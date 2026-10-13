#include "forge/frontend/onnx_parser.hpp"
#include "forge/core/error.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "onnx.pb.h"

namespace forge::frontend {

// ==========================================================
// Internal helpers
// ==========================================================
namespace {

ir::DataType convert_dtype(int onnx_dtype) {
    switch (onnx_dtype) {
        case onnx::TensorProto_DataType_FLOAT:   return ir::DataType::F32;
        case onnx::TensorProto_DataType_FLOAT16: return ir::DataType::F16;
        case onnx::TensorProto_DataType_BFLOAT16:return ir::DataType::BF16;
        case onnx::TensorProto_DataType_INT8:    return ir::DataType::I8;
        case onnx::TensorProto_DataType_INT16:   return ir::DataType::I16;
        case onnx::TensorProto_DataType_INT32:   return ir::DataType::I32;
        case onnx::TensorProto_DataType_INT64:   return ir::DataType::I64;
        case onnx::TensorProto_DataType_UINT8:   return ir::DataType::U8;
        default:                                  return ir::DataType::Unknown;
    }
}

ir::OpType convert_op(const std::string& op_type) {
    static const std::unordered_map<std::string, ir::OpType> MAP = {
        {"Add",       ir::OpType::Add},
        {"Sub",       ir::OpType::Sub},
        {"Mul",       ir::OpType::Mul},
        {"Div",       ir::OpType::Div},
        {"MatMul",    ir::OpType::MatMul},
        {"Gemm",      ir::OpType::Gemm},
        {"Conv",      ir::OpType::Conv2D},
        {"Relu",      ir::OpType::Relu},
        {"Sigmoid",   ir::OpType::Sigmoid},
        {"Tanh",      ir::OpType::Tanh},
        {"Softmax",   ir::OpType::Softmax},
        {"Gelu",      ir::OpType::Gelu},
        {"BatchNormalization",  ir::OpType::BatchNorm},
        {"LayerNormalization",  ir::OpType::LayerNorm},
        {"MaxPool",   ir::OpType::MaxPool},
        {"AveragePool", ir::OpType::AvgPool},
        {"GlobalAveragePool", ir::OpType::GlobalAvgPool},
        {"Reshape",   ir::OpType::Reshape},
        {"Flatten",   ir::OpType::Flatten},
        {"Transpose", ir::OpType::Transpose},
        {"Concat",    ir::OpType::Concat},
        {"Constant",  ir::OpType::Constant},
    };
    auto it = MAP.find(op_type);
    return it != MAP.end() ? it->second : ir::OpType::Unknown;
}

void convert_attributes(const onnx::NodeProto& onnx_node, ir::Node& node) {
    for (const auto& attr : onnx_node.attribute()) {
        const auto& name = attr.name();
        switch (attr.type()) {
            case onnx::AttributeProto_AttributeType_INT:
                node.attrs[name] = static_cast<std::int64_t>(attr.i());
                break;
            case onnx::AttributeProto_AttributeType_FLOAT:
                node.attrs[name] = static_cast<double>(attr.f());
                break;
            case onnx::AttributeProto_AttributeType_STRING:
                node.attrs[name] = attr.s();
                break;
            case onnx::AttributeProto_AttributeType_INTS: {
                std::vector<std::int64_t> v;
                v.reserve(attr.ints_size());
                for (int i = 0; i < attr.ints_size(); ++i)
                    v.push_back(static_cast<std::int64_t>(attr.ints(i)));
                node.attrs[name] = std::move(v);
                break;
            }
            case onnx::AttributeProto_AttributeType_FLOATS: {
                std::vector<double> v;
                v.reserve(attr.floats_size());
                for (int i = 0; i < attr.floats_size(); ++i)
                    v.push_back(static_cast<double>(attr.floats(i)));
                node.attrs[name] = std::move(v);
                break;
            }
            default: break;
        }
    }
}

ir::TensorInfo extract_tensor_info(const onnx::ValueInfoProto& vi) {
    ir::TensorInfo info;
    info.name = vi.name();

    if (!vi.has_type()) return info;
    const auto& type = vi.type();
    if (!type.has_tensor_type()) return info;

    const auto& tt = type.tensor_type();
    info.dtype = convert_dtype(tt.elem_type());

    if (tt.has_shape()) {
        const auto& shape = tt.shape();
        for (int i = 0; i < shape.dim_size(); ++i) {
            const auto& d = shape.dim(i);
            int64_t dim_value = 0;
            if (d.has_dim_value()) {
                dim_value = d.dim_value();
            } else if (d.has_dim_param()) {
                dim_value = -1;
            }
            info.shape.dims.push_back(dim_value);
        }
    }
    return info;
}

// ==========================================================
// Data extraction — NEW
// ==========================================================
std::vector<std::uint8_t> extract_raw_data(const onnx::TensorProto& t) {
    std::vector<std::uint8_t> out;

    // Case 1: raw_data provided (most common — pre-serialized bytes)
    if (!t.raw_data().empty()) {
        const auto& raw = t.raw_data();
        out.assign(raw.begin(), raw.end());
        return out;
    }

    // Case 2: typed arrays (older exporters)
    ir::DataType dtype = convert_dtype(t.data_type());

    std::size_t n = 1;
    for (int i = 0; i < t.dims_size(); ++i) {
        n *= static_cast<std::size_t>(t.dims(i));
    }
    if (n == 0) return out;

    switch (dtype) {
        case ir::DataType::F32: {
            out.resize(n * 4);
            for (std::size_t i = 0; i < n && i < (std::size_t)t.float_data_size(); ++i) {
                float v = t.float_data(static_cast<int>(i));
                std::memcpy(out.data() + i * 4, &v, 4);
            }
            break;
        }
        case ir::DataType::I32: {
            out.resize(n * 4);
            for (std::size_t i = 0; i < n && i < (std::size_t)t.int32_data_size(); ++i) {
                int32_t v = t.int32_data(static_cast<int>(i));
                std::memcpy(out.data() + i * 4, &v, 4);
            }
            break;
        }
        case ir::DataType::I64: {
            out.resize(n * 8);
            for (std::size_t i = 0; i < n && i < (std::size_t)t.int64_data_size(); ++i) {
                int64_t v = t.int64_data(static_cast<int>(i));
                std::memcpy(out.data() + i * 8, &v, 8);
            }
            break;
        }
        case ir::DataType::I8: {
            out.resize(n);
            for (std::size_t i = 0; i < n && i < (std::size_t)t.int32_data_size(); ++i) {
                out[i] = static_cast<std::uint8_t>(t.int32_data(static_cast<int>(i)));
            }
            break;
        }
        case ir::DataType::U8: {
            out.resize(n);
            for (std::size_t i = 0; i < n && i < (std::size_t)t.uint64_data_size(); ++i) {
                out[i] = static_cast<std::uint8_t>(t.uint64_data(static_cast<int>(i)));
            }
            break;
        }
        default:
            // Unsupported type — leave data empty
            break;
    }

    return out;
}

ir::Tensor extract_initializer_tensor(const onnx::TensorProto& t) {
    ir::TensorInfo info;
    info.name  = t.name();
    info.dtype = convert_dtype(t.data_type());
    for (int i = 0; i < t.dims_size(); ++i) {
        info.shape.dims.push_back(t.dims(i));
    }
    auto raw = extract_raw_data(t);
    return ir::Tensor(std::move(info), std::move(raw));
}

std::size_t extract_opset(const onnx::ModelProto& model) {
    std::size_t v = 0;
    for (int i = 0; i < model.opset_import_size(); ++i) {
        auto cur = static_cast<std::size_t>(model.opset_import(i).version());
        if (cur > v) v = cur;
    }
    return v;
}

} // anonymous namespace

// ==========================================================
// Public API
// ==========================================================

OnnxParseResult parse_onnx_bytes(const std::vector<std::uint8_t>& data) {
    if (data.empty()) {
        throw ParseError("Empty ONNX buffer");
    }

    onnx::ModelProto model;
    if (!model.ParseFromArray(data.data(), static_cast<int>(data.size()))) {
        throw ParseError("Failed to deserialize ONNX ModelProto (invalid binary)");
    }

    if (!model.has_graph()) {
        throw ParseError("ONNX ModelProto has no GraphProto");
    }

    const auto& g = model.graph();

    OnnxParseResult result;
    auto& graph    = result.graph;
    auto& metadata = result.metadata;

    metadata.producer_name     = model.producer_name();
    metadata.producer_version  = model.producer_version();
    metadata.domain            = model.domain();
    metadata.model_version     = model.model_version();
    metadata.ir_version        = model.ir_version();
    metadata.doc_string        = model.doc_string();
    metadata.opset_version     = extract_opset(model);
    metadata.node_count        = static_cast<std::size_t>(g.node_size());
    metadata.initializer_count = static_cast<std::size_t>(g.initializer_size());
    metadata.input_count       = static_cast<std::size_t>(g.input_size());
    metadata.output_count      = static_cast<std::size_t>(g.output_size());

    std::unordered_map<std::string, std::uint32_t> tensor_ids;

    auto register_info = [&](ir::TensorInfo info) -> std::uint32_t {
        auto id = graph.add_tensor(std::move(info));
        tensor_ids[graph.tensor(id)->name] = id;
        return id;
    };

    auto register_tensor = [&](ir::Tensor t) -> std::uint32_t {
        auto name = t.name();
        auto id = graph.add_tensor(std::move(t));
        tensor_ids[name] = id;
        return id;
    };

    // 1. Register initializers WITH DATA FIRST
    //    (Opset 7 lists weights as inputs — initializers must win)
    for (int i = 0; i < g.initializer_size(); ++i) {
        auto t = extract_initializer_tensor(g.initializer(i));
        if (tensor_ids.count(t.name())) continue;
        register_tensor(std::move(t));
    }

    // 2. Register graph inputs (metadata only, skip if already registered)
    for (int i = 0; i < g.input_size(); ++i) {
        auto info = extract_tensor_info(g.input(i));
        if (tensor_ids.count(info.name)) continue;
        register_info(std::move(info));
    }

    // 3. Register graph outputs
    for (int i = 0; i < g.output_size(); ++i) {
        auto info = extract_tensor_info(g.output(i));
        if (tensor_ids.count(info.name)) continue;
        register_info(std::move(info));
    }

    // 4. Create nodes
    for (int i = 0; i < g.node_size(); ++i) {
        const auto& onnx_node = g.node(i);

        ir::OpType op = convert_op(onnx_node.op_type());
        std::string name = onnx_node.name().empty()
            ? (onnx_node.op_type() + "_" + std::to_string(i))
            : onnx_node.name();

        auto node_id = graph.add_node(op, name);
        auto* node   = graph.node(node_id);

        convert_attributes(onnx_node, *node);

        auto ensure_tensor = [&](const std::string& tname) -> std::uint32_t {
            auto it = tensor_ids.find(tname);
            if (it != tensor_ids.end()) return it->second;
            ir::TensorInfo info;
            info.name = tname;
            info.dtype = ir::DataType::Unknown;
            return register_info(std::move(info));
        };

        for (int j = 0; j < onnx_node.input_size(); ++j)
            node->inputs.push_back(ensure_tensor(onnx_node.input(j)));
        for (int j = 0; j < onnx_node.output_size(); ++j)
            node->outputs.push_back(ensure_tensor(onnx_node.output(j)));
    }

    return result;
}

OnnxParseResult parse_onnx(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ParseError("Cannot open file: " + path.string());
    }

    in.seekg(0, std::ios::end);
    auto size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size <= 0) {
        throw ParseError("File is empty: " + path.string());
    }

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(size));
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw ParseError("Failed to read: " + path.string());
    }

    return parse_onnx_bytes(buffer);
}

} // namespace forge::frontend
