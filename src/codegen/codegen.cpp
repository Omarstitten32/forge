#include "forge/codegen/codegen.hpp"
#include "forge/ir/op.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge::codegen {

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::NEON:   return "neon";
        case Backend::SVE:    return "sve";
        case Backend::APU:    return "apu";
        case Backend::Vulkan: return "vulkan";
    }
    return "unknown";
}

namespace {

std::size_t elem_count(const ir::Shape& s) {
    if (s.dims.empty()) return 0;
    std::size_t n = 1;
    for (auto d : s.dims) { if (d <= 0) return 0; n *= std::size_t(d); }
    return n;
}

std::int64_t attr_i64(const ir::Node& n, const char* k, std::int64_t fb) {
    auto it = n.attrs.find(k);
    if (it == n.attrs.end()) return fb;
    if (auto p = std::get_if<std::int64_t>(&it->second)) return *p;
    return fb;
}

std::vector<std::int64_t> attr_i64_list(const ir::Node& n, const char* k) {
    auto it = n.attrs.find(k);
    if (it == n.attrs.end()) return {};
    if (auto p = std::get_if<std::vector<std::int64_t>>(&it->second)) return *p;
    return {};
}

class Emitter {
public:
    Emitter(const ir::Graph& g, Backend backend) : g_(g), backend_(backend) {}

    GeneratedCode emit() {
        scan_roles();
        layout_weights();
        layout_intermediates();
        emit_header();
        emit_weight_offsets();
        emit_run_function();
        GeneratedCode out;
        out.source  = out_.str();
        out.weights = std::move(weights_blob_);
        out.signature = std::string("backend=") + backend_name(backend_) +
                        " nodes=" + std::to_string(g_.node_count()) +
                        " weights_bytes=" + std::to_string(out.weights.size());
        return out;
    }

private:
    const ir::Graph& g_;
    Backend backend_;
    std::ostringstream out_;
    std::vector<std::uint8_t> weights_blob_;
    std::unordered_map<std::uint32_t, std::size_t> weight_offset_;
    std::unordered_map<std::uint32_t, std::size_t> intermediate_offset_;
    std::vector<std::uint32_t> graph_inputs_;
    std::vector<std::uint32_t> graph_outputs_;
    std::size_t intermediate_total_{0};

    void scan_roles() {
        std::vector<bool> consumed(g_.tensors().size(), false);
        for (const auto& n : g_.nodes())
            for (auto in : n.inputs)
                if (in < consumed.size()) consumed[in] = true;
        for (std::size_t i = 0; i < g_.tensors().size(); ++i) {
            const auto& t = g_.tensors()[i];
            auto id = static_cast<std::uint32_t>(i);
            if (t.name() == "input" || t.name() == "data" || t.name() == "X")
                graph_inputs_.push_back(id);
            if (!t.has_data() && !consumed[i] && !t.shape().dims.empty() &&
                t.name() != "input" && t.name() != "data" && t.name() != "X" &&
                t.name() != "__dead__")
                graph_outputs_.push_back(id);
        }
    }

    void layout_weights() {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < g_.tensors().size(); ++i) {
            const auto* t = g_.tensor_data(static_cast<std::uint32_t>(i));
            if (!t || !t->has_data()) continue;
            std::size_t n = elem_count(t->shape());
            if (n == 0) continue;
            weight_offset_[static_cast<std::uint32_t>(i)] = offset;
            if (t->dtype() == ir::DataType::F32) {
                const std::uint8_t* raw =
                    reinterpret_cast<const std::uint8_t*>(t->as<float>());
                weights_blob_.insert(weights_blob_.end(), raw, raw + n * 4);
            } else {
                weights_blob_.insert(weights_blob_.end(), n * 4, 0);
            }
            offset += n;
        }
    }

    void layout_intermediates() {
        // Iterate over ALL tensors, not just node outputs.
        // After fusion, many intermediates are no longer node outputs
        // but still referenced as inputs by other nodes.
        for (std::size_t i = 0; i < g_.tensors().size(); ++i) {
            std::uint32_t tid = static_cast<std::uint32_t>(i);
            if (weight_offset_.count(tid)) continue;
            bool skip = false;
            for (auto x : graph_inputs_)  if (x == tid) { skip = true; break; }
            for (auto x : graph_outputs_) if (x == tid) { skip = true; break; }
            if (skip) continue;
            const auto* t = g_.tensor(tid);
            if (!t) continue;
            std::size_t n = elem_count(t->shape);
            if (n == 0) continue;
            intermediate_offset_[tid] = intermediate_total_;
            intermediate_total_ += n;
        }
    }

    std::string weight_ptr(std::uint32_t tid) const {
        if (weight_offset_.count(tid)) return "W_" + std::to_string(tid);
        for (auto x : graph_inputs_)  if (x == tid) return "input";
        for (auto x : graph_outputs_) if (x == tid) return "output";
        // Fallback: return a valid pointer via a bound local. If the tensor
        // wasn't laid out, it's a codegen bug — return nullptr so it crashes
        // at the call site instead of emitting an undeclared identifier.
        if (!intermediate_offset_.count(tid))
            return "nullptr /* codegen bug: t_" + std::to_string(tid) + " unallocated */";
        return "t_" + std::to_string(tid);
    }

    void emit_header() {
        out_ << "// Auto-generated by Forge v0.1.0\n";
        out_ << "// backend: " << backend_name(backend_) << "\n";
        out_ << "// " << g_.summary() << "\n\n";
        out_ << "#include <cstring>\n#include <vector>\n\n";
        out_ << "#include \"forge/codegen/kernels.hpp\"\n\n";
    }

    void emit_weight_offsets() {
        out_ << "// Weight offsets (float indices into `weights`)\n";
        std::vector<std::pair<std::uint32_t, std::size_t>> items(
            weight_offset_.begin(), weight_offset_.end());
        std::sort(items.begin(), items.end());
        for (auto& [tid, off] : items)
            out_ << "static const std::size_t OFF_W_" << tid << " = " << off << ";\n";
        out_ << "\n// Intermediate offsets\n";
        std::vector<std::pair<std::uint32_t, std::size_t>> items2(
            intermediate_offset_.begin(), intermediate_offset_.end());
        std::sort(items2.begin(), items2.end());
        for (auto& [tid, off] : items2)
            out_ << "static const std::size_t OFF_T_" << tid << " = " << off << ";\n";
        out_ << "static const std::size_t INTERMEDIATE_TOTAL = "
             << intermediate_total_ << ";\n\n";
    }

    void emit_conv(const ir::Node& n, std::ostringstream& b,
                   const std::string& dst, std::uint32_t in_tid,
                   std::uint32_t w_tid, std::int64_t bias_tid) {
        const auto* in = g_.tensor(in_tid);
        const auto* w  = g_.tensor(w_tid);
        if (!in || !w) return;
        auto ids = in->shape.dims;
        auto wds = w->shape.dims;
        if (ids.size() != 4 || wds.size() != 4) return;
        std::int64_t N=ids[0], Cin=ids[1], H=ids[2], W=ids[3];
        std::int64_t Cout=wds[0], KH=wds[2], KW=wds[3];
        auto strides = attr_i64_list(n, "strides");
        auto pads    = attr_i64_list(n, "pads");
        auto kshape  = attr_i64_list(n, "kernel_shape");
        std::int64_t sh = strides.size()>=2 ? strides[0] : 1;
        std::int64_t sw = strides.size()>=2 ? strides[1] : 1;
        std::int64_t pt = pads.size()>=4    ? pads[0]    : 0;
        std::int64_t pl = pads.size()>=4    ? pads[1]    : 0;
        if (kshape.size()==2) { KH=kshape[0]; KW=kshape[1]; }
        const auto* ot = g_.tensor(n.outputs[0]);
        std::int64_t H_out = ot ? ot->shape.dims[2] : (H + 2*pt - KH)/sh + 1;
        std::int64_t W_out = ot ? ot->shape.dims[3] : (W + 2*pl - KW)/sw + 1;
        std::string bias = (bias_tid >= 0)
            ? weight_ptr(static_cast<std::uint32_t>(bias_tid)) : "nullptr";
        if ((wds[1] == 1) && (Cin == Cout) && (Cin > 1)) {
            b << "  forge::codegen::conv2d_depthwise_f32(" << weight_ptr(in_tid) << ","
              << weight_ptr(w_tid) << "," << bias << "," << dst << ","
              << N << "," << Cin << "," << H << "," << W << ","
              << KH << "," << KW << ","
              << sh << "," << sw << "," << pt << "," << pl << ","
              << H_out << "," << W_out << ");\n";
        } else {
            b << "  forge::codegen::conv2d_f32_naive(" << weight_ptr(in_tid) << ","
              << weight_ptr(w_tid) << "," << bias << "," << dst << ","
              << N << "," << Cin << "," << H << "," << W << ","
              << Cout << "," << KH << "," << KW << ","
              << sh << "," << sw << "," << pt << "," << pl << ","
              << H_out << "," << W_out << ");\n";
        }
    }

    void emit_bn(const ir::Node& n, std::ostringstream& b, const std::string& dst,
                 std::uint32_t in_tid, std::uint32_t g_tid, std::uint32_t be_tid,
                 std::uint32_t m_tid, std::uint32_t v_tid) {
        const auto* in = g_.tensor(in_tid);
        if (!in) return;
        auto d = in->shape.dims;
        if (d.size() != 4) return;
        float eps = 1e-5f;
        auto it = n.attrs.find("epsilon");
        if (it != n.attrs.end())
            if (auto p = std::get_if<double>(&it->second)) eps = float(*p);
        b << "  forge::codegen::batchnorm_f32(" << weight_ptr(in_tid) << ","
          << weight_ptr(g_tid) << "," << weight_ptr(be_tid) << ","
          << weight_ptr(m_tid) << "," << weight_ptr(v_tid) << "," << dst << ","
          << d[0]<<","<<d[1]<<","<<d[2]<<","<<d[3]<<"," << eps << "f);\n";
    }

    void emit_node(const ir::Node& n, std::ostringstream& b) {
        if (n.op == ir::OpType::Unknown) { b << "  // skipped Unknown\n"; return; }
        if (n.outputs.empty()) return;
        const std::string dst = weight_ptr(n.outputs[0]);
        const auto* out_t = g_.tensor(n.outputs[0]);
        const std::size_t out_n = out_t ? elem_count(out_t->shape) : 0;

        switch (n.op) {
            case ir::OpType::Conv2D: {
                std::int64_t bias = n.inputs.size()>=3 ? (std::int64_t)n.inputs[2] : -1;
                emit_conv(n, b, dst, n.inputs[0], n.inputs[1], bias);
                break;
            }
            case ir::OpType::Conv2DRelu: {
                std::int64_t bias = n.inputs.size()>=3 ? (std::int64_t)n.inputs[2] : -1;
                emit_conv(n, b, dst, n.inputs[0], n.inputs[1], bias);
                b << "  forge::codegen::relu_f32(" << dst << "," << dst << ","
                  << out_n << ");\n";
                break;
            }
            case ir::OpType::Conv2DBatchNorm:
            case ir::OpType::Conv2DBatchNormRelu: {
                std::int64_t w_i = n.inputs.size()>=2 ? n.inputs[1] : -1;
                std::int64_t b_i=-1, g_i=-1, be_i=-1, m_i=-1, v_i=-1;
                // Two layouts possible:
                //   7 inputs: [in, w, bias, gamma, beta, mean, var]
                //   6 inputs: [in, w, gamma, beta, mean, var]  (no conv bias)
                if (n.inputs.size() >= 7) {
                    b_i=n.inputs[2]; g_i=n.inputs[3];
                    be_i=n.inputs[4]; m_i=n.inputs[5]; v_i=n.inputs[6];
                } else if (n.inputs.size() == 6) {
                    b_i=-1; g_i=n.inputs[2];
                    be_i=n.inputs[3]; m_i=n.inputs[4]; v_i=n.inputs[5];
                }
                if (w_i < 0) break;
                emit_conv(n, b, dst, n.inputs[0], (std::uint32_t)w_i, b_i);
                if (g_i >= 0)
                    emit_bn(n, b, dst, n.outputs[0],
                            (std::uint32_t)g_i, (std::uint32_t)be_i,
                            (std::uint32_t)m_i, (std::uint32_t)v_i);
                if (n.op == ir::OpType::Conv2DBatchNormRelu)
                    b << "  forge::codegen::relu_f32(" << dst << "," << dst << ","
                      << out_n << ");\n";
                break;
            }
            case ir::OpType::Gemm:
            case ir::OpType::MatMul: {
                const auto* A = g_.tensor(n.inputs[0]);
                const auto* B = g_.tensor(n.inputs[1]);
                if (!A || !B) break;
                auto ad = A->shape.dims, bd = B->shape.dims;
                if (ad.size()!=2 || bd.size()!=2) break;
                bool tA = attr_i64(n, "transA", 0) != 0;
                bool tB = attr_i64(n, "transB", 0) != 0;
                std::int64_t M = tA ? ad[1] : ad[0];
                std::int64_t K = tA ? ad[0] : ad[1];
                std::int64_t Nn = tB ? bd[0] : bd[1];
                std::string bias = n.inputs.size()>=3 ? weight_ptr(n.inputs[2]) : "nullptr";
                b << "  std::memset(" << dst << ",0," << (M*Nn*4) << ");\n";
                b << "  forge::codegen::gemm_f32_naive("
                  << weight_ptr(n.inputs[0]) << "," << weight_ptr(n.inputs[1]) << ","
                  << bias << "," << dst << ","
                  << M << "," << K << "," << Nn << ",1.0f,0.0f,"
                  << (tA?"true":"false") << "," << (tB?"true":"false") << ");\n";
                break;
            }
            case ir::OpType::Relu:
                b << "  forge::codegen::relu_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << out_n << ");\n"; break;
            case ir::OpType::Sigmoid:
                b << "  forge::codegen::sigmoid_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << out_n << ");\n"; break;
            case ir::OpType::Tanh:
                b << "  forge::codegen::tanh_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << out_n << ");\n"; break;
            case ir::OpType::Gelu:
                b << "  forge::codegen::gelu_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << out_n << ");\n"; break;
            case ir::OpType::Add:
                b << "  forge::codegen::add_f32(" << weight_ptr(n.inputs[0]) << ","
                  << weight_ptr(n.inputs[1]) << "," << dst << "," << out_n << ");\n"; break;
            case ir::OpType::AddRelu:
                b << "  forge::codegen::add_f32(" << weight_ptr(n.inputs[0]) << ","
                  << weight_ptr(n.inputs[1]) << "," << dst << "," << out_n << ");\n"
                  << "  forge::codegen::relu_f32(" << dst << "," << dst << ","
                  << out_n << ");\n"; break;
            case ir::OpType::Mul:
                b << "  forge::codegen::mul_f32(" << weight_ptr(n.inputs[0]) << ","
                  << weight_ptr(n.inputs[1]) << "," << dst << "," << out_n << ");\n"; break;
            case ir::OpType::BatchNorm:
            case ir::OpType::LayerNorm: {
                std::uint32_t g_i = n.inputs.size()>1 ? n.inputs[1] : n.inputs[0];
                std::uint32_t b_i = n.inputs.size()>2 ? n.inputs[2] : n.inputs[0];
                std::uint32_t m_i = n.inputs.size()>3 ? n.inputs[3] : n.inputs[0];
                std::uint32_t v_i = n.inputs.size()>4 ? n.inputs[4] : n.inputs[0];
                emit_bn(n, b, dst, n.inputs[0], g_i, b_i, m_i, v_i);
                break;
            }
            case ir::OpType::MaxPool: {
                const auto* in = g_.tensor(n.inputs[0]);
                const auto* ot = g_.tensor(n.outputs[0]);
                if (!in || !ot) break;
                auto d = in->shape.dims; auto od = ot->shape.dims;
                if (d.size()!=4 || od.size()!=4) break;
                auto ks = attr_i64_list(n, "kernel_shape");
                auto st = attr_i64_list(n, "strides");
                std::int64_t KH = ks.size()>=2 ? ks[0] : d[2];
                std::int64_t KW = ks.size()>=2 ? ks[1] : d[3];
                std::int64_t sh = st.size()>=2 ? st[0] : 1;
                std::int64_t sw = st.size()>=2 ? st[1] : 1;
                b << "  forge::codegen::maxpool2d_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << d[0]<<","<<d[1]<<","<<d[2]<<","<<d[3]<<","
                  << KH<<","<<KW<<","<<sh<<","<<sw<<","
                  << od[2]<<","<<od[3] << ");\n"; break;
            }
            case ir::OpType::AvgPool:
                b << "  // TODO AvgPool\n"; break;
            case ir::OpType::GlobalAvgPool: {
                const auto* in = g_.tensor(n.inputs[0]);
                if (!in) break;
                auto d = in->shape.dims;
                if (d.size()!=4) break;
                b << "  forge::codegen::globalavgpool_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << d[0]<<","<<d[1]<<","<<d[2]<<","<<d[3] << ");\n"; break;
            }
            case ir::OpType::Flatten:
                b << "  forge::codegen::flatten_f32(" << weight_ptr(n.inputs[0]) << ","
                  << dst << "," << out_n << ");\n"; break;
            case ir::OpType::Reshape:
                b << "  std::memcpy(" << dst << "," << weight_ptr(n.inputs[0]) << ","
                  << (out_n * 4) << ");\n"; break;
            case ir::OpType::Constant:
                b << "  // Constant\n"; break;
            default:
                b << "  // unsupported: " << ir::op_name(n.op) << "\n"; break;
        }
    }

    void emit_run_function() {
        std::vector<std::pair<std::uint32_t, std::size_t>> items(
            weight_offset_.begin(), weight_offset_.end());
        std::sort(items.begin(), items.end());

        out_ << "extern \"C\" void forge_run(const float* input, float* output,\n"
                "                          const float* weights)\n{\n";

        for (auto& [tid, off] : items)
            out_ << "  const float* W_" << tid << " = weights + " << off << ";\n";

        if (intermediate_total_ > 0) {
            out_ << "  std::vector<float> tbuf(" << intermediate_total_ << ");\n";
            out_ << "  float* tbase = tbuf.data();\n";
            std::vector<std::pair<std::uint32_t, std::size_t>> items2(
                intermediate_offset_.begin(), intermediate_offset_.end());
            std::sort(items2.begin(), items2.end());
            for (auto& [tid, off] : items2)
                out_ << "  float* t_" << tid << " = tbase + " << off << ";\n";
        }
        out_ << "\n";

        for (const auto& n : g_.nodes()) {
            if (n.name == "__dead__") continue;
            out_ << "  // " << ir::op_name(n.op) << "  " << n.name << "\n";
            std::ostringstream body;
            emit_node(n, body);
            out_ << body.str() << "\n";
        }
        out_ << "}\n";
    }
};

} // namespace

GeneratedCode generate(const ir::Graph& graph, Backend backend) {
    Emitter e(graph, backend);
    return e.emit();
}

} // namespace forge::codegen
