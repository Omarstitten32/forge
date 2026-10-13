#include "forge/ir/shape_infer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace forge::ir {
namespace {

std::int64_t attr_int(const Node& n, const char* key, std::int64_t fb) {
    auto it = n.attrs.find(key);
    if (it == n.attrs.end()) return fb;
    if (auto p = std::get_if<std::int64_t>(&it->second)) return *p;
    return fb;
}

std::vector<std::int64_t> attr_int_list(const Node& n, const char* key) {
    auto it = n.attrs.find(key);
    if (it == n.attrs.end()) return {};
    if (auto p = std::get_if<std::vector<std::int64_t>>(&it->second)) return *p;
    return {};
}

bool has_shape(const TensorInfo* info) {
    return info && !info->shape.dims.empty();
}

TensorInfo make_info(const std::string& name, DataType dt,
                     std::vector<std::int64_t> dims) {
    TensorInfo info;
    info.name  = name;
    info.dtype = dt;
    info.shape.dims = std::move(dims);
    return info;
}

std::vector<std::uint32_t> topo_order(const Graph& g) {
    std::unordered_map<std::uint32_t, std::uint32_t> producer;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> consumers;
    std::unordered_map<std::uint32_t, const Node*> by_id;
    std::unordered_map<std::uint32_t, std::size_t> indeg;

    for (const auto& n : g.nodes()) {
        by_id[n.id] = &n;
        indeg[n.id] = 0;
        for (auto t : n.outputs) producer[t] = n.id;
        for (auto t : n.inputs)  consumers[t].push_back(n.id);
    }
    for (const auto& n : g.nodes())
        for (auto t : n.inputs)
            if (producer.count(t)) indeg[n.id]++;

    std::queue<std::uint32_t> q;
    for (const auto& n : g.nodes())
        if (indeg[n.id] == 0) q.push(n.id);

    std::vector<std::uint32_t> order;
    order.reserve(g.node_count());
    while (!q.empty()) {
        auto id = q.front(); q.pop();
        order.push_back(id);
        auto it = by_id.find(id);
        if (it == by_id.end()) continue;
        for (auto t : it->second->outputs) {
            auto ci = consumers.find(t);
            if (ci == consumers.end()) continue;
            for (auto cid : ci->second)
                if (--indeg[cid] == 0) q.push(cid);
        }
    }
    return order;
}

bool infer_identity(const Node& n, Graph& g) {
    if (n.inputs.empty() || n.outputs.empty()) return false;
    const auto* in = g.tensor(n.inputs[0]);
    if (!has_shape(in)) return false;
    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), in->dtype, in->shape.dims));
    return true;
}

bool infer_conv(const Node& n, Graph& g) {
    if (n.inputs.size() < 2 || n.outputs.empty()) return false;
    const auto* in = g.tensor(n.inputs[0]);
    const auto* w  = g.tensor(n.inputs[1]);
    if (!has_shape(in) || !has_shape(w)) return false;
    if (in->shape.dims.size() != 4 || w->shape.dims.size() != 4) return false;

    const auto& id_ = in->shape.dims;
    const auto& wd  = w->shape.dims;
    std::int64_t N = id_[0], H = id_[2], W = id_[3];
    std::int64_t C_out = wd[0], KH = wd[2], KW = wd[3];

    auto strides = attr_int_list(n, "strides");
    auto pads    = attr_int_list(n, "pads");
    auto kshape  = attr_int_list(n, "kernel_shape");

    std::int64_t sh = (strides.size() >= 2) ? strides[0] : 1;
    std::int64_t sw = (strides.size() >= 2) ? strides[1] : 1;
    std::int64_t pt = (pads.size()    >= 4) ? pads[0]    : 0;
    std::int64_t pl = (pads.size()    >= 4) ? pads[1]    : 0;
    std::int64_t pb = (pads.size()    >= 4) ? pads[2]    : 0;
    std::int64_t pr = (pads.size()    >= 4) ? pads[3]    : 0;
    if (kshape.size() == 2) { KH = kshape[0]; KW = kshape[1]; }

    std::int64_t H_out = (H + pt + pb - KH) / sh + 1;
    std::int64_t W_out = (W + pl + pr - KW) / sw + 1;

    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), in->dtype, {N, C_out, H_out, W_out}));
    return true;
}

bool infer_pool(const Node& n, Graph& g) {
    if (n.inputs.empty() || n.outputs.empty()) return false;
    const auto* in = g.tensor(n.inputs[0]);
    if (!has_shape(in) || in->shape.dims.size() != 4) return false;
    const auto& d = in->shape.dims;
    std::int64_t N = d[0], C = d[1], H = d[2], W = d[3];

    auto kshape  = attr_int_list(n, "kernel_shape");
    auto strides = attr_int_list(n, "strides");
    auto pads    = attr_int_list(n, "pads");

    std::int64_t KH = (kshape.size()  >= 2) ? kshape[0]  : H;
    std::int64_t KW = (kshape.size()  >= 2) ? kshape[1]  : W;
    std::int64_t sh = (strides.size() >= 2) ? strides[0] : 1;
    std::int64_t sw = (strides.size() >= 2) ? strides[1] : 1;
    std::int64_t pt = (pads.size()    >= 4) ? pads[0]    : 0;
    std::int64_t pl = (pads.size()    >= 4) ? pads[1]    : 0;
    std::int64_t pb = (pads.size()    >= 4) ? pads[2]    : 0;
    std::int64_t pr = (pads.size()    >= 4) ? pads[3]    : 0;

    std::int64_t H_out = (H + pt + pb - KH) / sh + 1;
    std::int64_t W_out = (W + pl + pr - KW) / sw + 1;

    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), in->dtype, {N, C, H_out, W_out}));
    return true;
}

bool infer_global_avgpool(const Node& n, Graph& g) {
    if (n.inputs.empty() || n.outputs.empty()) return false;
    const auto* in = g.tensor(n.inputs[0]);
    if (!has_shape(in) || in->shape.dims.size() != 4) return false;
    const auto& d = in->shape.dims;
    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), in->dtype, {d[0], d[1], 1, 1}));
    return true;
}

bool infer_gemm(const Node& n, Graph& g) {
    if (n.inputs.size() < 2 || n.outputs.empty()) return false;
    const auto* A = g.tensor(n.inputs[0]);
    const auto* B = g.tensor(n.inputs[1]);
    if (!has_shape(A) || !has_shape(B)) return false;
    if (A->shape.dims.size() != 2 || B->shape.dims.size() != 2) return false;
    bool transA = attr_int(n, "transA", 0) != 0;
    bool transB = attr_int(n, "transB", 0) != 0;
    std::int64_t M = transA ? A->shape.dims[1] : A->shape.dims[0];
    std::int64_t N = transB ? B->shape.dims[0] : B->shape.dims[1];
    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), A->dtype, {M, N}));
    return true;
}

bool infer_broadcast(const Node& n, Graph& g) {
    if (n.inputs.size() < 2 || n.outputs.empty()) return false;
    const auto* a = g.tensor(n.inputs[0]);
    const auto* b = g.tensor(n.inputs[1]);
    if (!has_shape(a) || !has_shape(b)) return false;
    const auto& da = a->shape.dims;
    const auto& db = b->shape.dims;
    if (da == db) {
        auto* out = g.tensor_data(n.outputs[0]);
        if (!out) return false;
        out->set_info(make_info(out->name(), a->dtype, da));
        return true;
    }
    std::size_t rank = std::max(da.size(), db.size());
    std::vector<std::int64_t> result(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        std::size_t ia = i + da.size() - rank;
        std::size_t ib = i + db.size() - rank;
        std::int64_t va = (i >= rank - da.size()) ? da[ia] : 1;
        std::int64_t vb = (i >= rank - db.size()) ? db[ib] : 1;
        if      (va == vb) result[i] = va;
        else if (va == 1)  result[i] = vb;
        else if (vb == 1)  result[i] = va;
        else return false;
    }
    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), a->dtype, std::move(result)));
    return true;
}

bool infer_flatten(const Node& n, Graph& g) {
    if (n.inputs.empty() || n.outputs.empty()) return false;
    const auto* in = g.tensor(n.inputs[0]);
    if (!has_shape(in)) return false;
    std::int64_t axis = attr_int(n, "axis", 1);
    const auto& d = in->shape.dims;
    if (axis < 0) axis += static_cast<std::int64_t>(d.size());
    if (axis < 0 || axis > static_cast<std::int64_t>(d.size())) return false;
    std::int64_t outer = 1, inner = 1;
    for (std::int64_t i = 0; i < axis; ++i) outer *= d[static_cast<std::size_t>(i)];
    for (std::size_t i = static_cast<std::size_t>(axis); i < d.size(); ++i) inner *= d[i];
    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), in->dtype, {outer, inner}));
    return true;
}

bool infer_reshape(const Node& n, Graph& g) {
    if (n.inputs.size() < 2 || n.outputs.empty()) return false;
    const auto* shape_t = g.tensor_data(n.inputs[1]);
    if (!shape_t || !shape_t->has_data()) return false;
    if (shape_t->dtype() != DataType::I64) return false;
    const std::size_t count = shape_t->element_count();
    if (count == 0) return false;
    const auto* raw = shape_t->as<std::int64_t>();
    if (!raw) return false;

    std::vector<std::int64_t> req(raw, raw + count);

    // Input shape (may be missing)
    const auto* in_info = g.tensor(n.inputs[0]);
    std::vector<std::int64_t> in_dims;
    std::int64_t in_total = 1;
    bool in_known = false;
    if (in_info && !in_info->shape.dims.empty()) {
        in_dims = in_info->shape.dims;
        for (auto d : in_dims) in_total *= d;
        in_known = (in_total > 0);
    }

    // Resolve 0 and -1 per ONNX Reshape semantics
    std::vector<std::int64_t> out_dims(req.size());
    std::int64_t minus_one_idx = -1;
    std::int64_t known_prod = 1;
    for (std::size_t i = 0; i < req.size(); ++i) {
        std::int64_t v = req[i];
        if (v == 0) {
            if (!in_known || i >= in_dims.size()) return false;
            v = in_dims[i];
        } else if (v == -1) {
            if (minus_one_idx >= 0) return false;
            minus_one_idx = static_cast<std::int64_t>(i);
            v = 1;
        }
        if (v <= 0) return false;
        out_dims[i] = v;
        known_prod *= v;
    }

    if (minus_one_idx >= 0) {
        if (!in_known || known_prod <= 0) return false;
        if (in_total % known_prod != 0) return false;
        out_dims[minus_one_idx] = in_total / known_prod;
    } else if (in_known && in_total != known_prod) {
        return false;
    }

    DataType dt = DataType::F32;
    if (in_info && in_info->dtype != DataType::Unknown) dt = in_info->dtype;

    auto* out = g.tensor_data(n.outputs[0]);
    if (!out) return false;
    out->set_info(make_info(out->name(), dt, std::move(out_dims)));
    return true;
}

} // namespace

ShapeInferenceStats infer_shapes(Graph& g) {
    ShapeInferenceStats stats;
    auto order = topo_order(g);
    for (auto nid : order) {
        auto* n = g.node(nid);
        if (!n) continue;
        stats.nodes_processed++;
        bool ok = false;
        switch (n->op) {
            case OpType::Conv2D:                                  ok = infer_conv(*n, g);           break;
            case OpType::BatchNorm:
            case OpType::LayerNorm:
            case OpType::Relu:
            case OpType::Sigmoid:
            case OpType::Tanh:
            case OpType::Gelu:
            case OpType::Softmax:                                 ok = infer_identity(*n, g);       break;
            case OpType::MaxPool:
            case OpType::AvgPool:                                 ok = infer_pool(*n, g);           break;
            case OpType::GlobalAvgPool:                           ok = infer_global_avgpool(*n,g);  break;
            case OpType::Gemm:
            case OpType::MatMul:                                  ok = infer_gemm(*n, g);           break;
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            case OpType::Div:                                     ok = infer_broadcast(*n, g);      break;
            case OpType::Flatten:                                 ok = infer_flatten(*n, g);        break;
            case OpType::Reshape:                                 ok = infer_reshape(*n, g);        break;
            case OpType::Constant:                                ok = true;                        break;
            default:                                              ok = false;                       break;
        }
        if (ok) {
            stats.tensors_inferred++;
        } else {
            stats.nodes_skipped++;
            std::fprintf(stderr,
                "[shape-infer] skipped: op=%s node_id=%u name=%s\n",
                op_name(n->op), n->id, n->name.c_str());
        }
    }
    return stats;
}

} // namespace forge::ir
