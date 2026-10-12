#include "forge/optimizer/evaluate.hpp"
#include "forge/core/error.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace forge::optimizer {

// ==========================================================
// Helpers
// ==========================================================
namespace {

/// Copy info from a source tensor but keep the given name.
ir::TensorInfo rename_info(const ir::Tensor& t, const std::string& name) {
    ir::TensorInfo info = t.info();
    info.name = name;
    return info;
}

/// Verify shapes match element-wise.
bool shapes_equal(const ir::Tensor& a, const ir::Tensor& b) {
    return a.shape().dims == b.shape().dims;
}

/// Verify a tensor is F32 with data.
bool is_f32_with_data(const ir::Tensor& t) {
    return t.has_data() && t.dtype() == ir::DataType::F32;
}

} // anonymous namespace

// ==========================================================
// Element-wise binary operations
// ==========================================================
#define DEFINE_BINARY_OP(name, OP)                                              \
ir::Tensor name(const ir::Tensor& a, const ir::Tensor& b) {                     \
    if (!is_f32_with_data(a) || !is_f32_with_data(b)) {                         \
        throw OptimizationError(#name ": requires F32 with data");              \
    }                                                                            \
    if (!shapes_equal(a, b)) {                                                   \
        throw OptimizationError(#name ": shape mismatch");                      \
    }                                                                            \
    std::size_t n = a.element_count();                                           \
    std::vector<std::uint8_t> out(n * 4);                                        \
    const float* pa = a.as<float>();                                             \
    const float* pb = b.as<float>();                                             \
    float*       po = reinterpret_cast<float*>(out.data());                      \
    for (std::size_t i = 0; i < n; ++i) {                                        \
        po[i] = pa[i] OP pb[i];                                                  \
    }                                                                            \
    return ir::Tensor(a.info(), std::move(out));                                 \
}

DEFINE_BINARY_OP(eval_add, +)
DEFINE_BINARY_OP(eval_sub, -)
DEFINE_BINARY_OP(eval_mul, *)
DEFINE_BINARY_OP(eval_div, /)

#undef DEFINE_BINARY_OP

// ==========================================================
// Unary operations
// ==========================================================
ir::Tensor eval_relu(const ir::Tensor& a) {
    if (!is_f32_with_data(a)) {
        throw OptimizationError("eval_relu: requires F32 with data");
    }
    std::size_t n = a.element_count();
    std::vector<std::uint8_t> out(n * 4);
    const float* pa = a.as<float>();
    float*       po = reinterpret_cast<float*>(out.data());
    for (std::size_t i = 0; i < n; ++i) {
        po[i] = pa[i] > 0.0f ? pa[i] : 0.0f;
    }
    return ir::Tensor(a.info(), std::move(out));
}

ir::Tensor eval_sigmoid(const ir::Tensor& a) {
    if (!is_f32_with_data(a)) {
        throw OptimizationError("eval_sigmoid: requires F32 with data");
    }
    std::size_t n = a.element_count();
    std::vector<std::uint8_t> out(n * 4);
    const float* pa = a.as<float>();
    float*       po = reinterpret_cast<float*>(out.data());
    for (std::size_t i = 0; i < n; ++i) {
        po[i] = 1.0f / (1.0f + std::exp(-pa[i]));
    }
    return ir::Tensor(a.info(), std::move(out));
}

ir::Tensor eval_tanh(const ir::Tensor& a) {
    if (!is_f32_with_data(a)) {
        throw OptimizationError("eval_tanh: requires F32 with data");
    }
    std::size_t n = a.element_count();
    std::vector<std::uint8_t> out(n * 4);
    const float* pa = a.as<float>();
    float*       po = reinterpret_cast<float*>(out.data());
    for (std::size_t i = 0; i < n; ++i) {
        po[i] = std::tanh(pa[i]);
    }
    return ir::Tensor(a.info(), std::move(out));
}

// ==========================================================
// Generic evaluator
// ==========================================================
std::optional<ir::Tensor> try_evaluate(
    const ir::Node& node,
    const std::vector<const ir::Tensor*>& inputs,
    const ir::TensorInfo& output_info
) {
    // All inputs must have data
    for (auto* t : inputs) {
        if (!t || !t->has_data()) return std::nullopt;
    }

    auto rename = [&](ir::Tensor t) {
        ir::TensorInfo info = output_info;
        // Preserve computed dtype/shape if it matches
        info.dtype = t.dtype();
        info.shape = t.shape();
        return ir::Tensor(std::move(info), std::move(t).bytes());
    };

    try {
        switch (node.op) {
            case ir::OpType::Add:
                if (inputs.size() == 2) return rename(eval_add(*inputs[0], *inputs[1]));
                break;
            case ir::OpType::Sub:
                if (inputs.size() == 2) return rename(eval_sub(*inputs[0], *inputs[1]));
                break;
            case ir::OpType::Mul:
                if (inputs.size() == 2) return rename(eval_mul(*inputs[0], *inputs[1]));
                break;
            case ir::OpType::Div:
                if (inputs.size() == 2) return rename(eval_div(*inputs[0], *inputs[1]));
                break;
            case ir::OpType::Relu:
                if (inputs.size() == 1) return rename(eval_relu(*inputs[0]));
                break;
            case ir::OpType::Sigmoid:
                if (inputs.size() == 1) return rename(eval_sigmoid(*inputs[0]));
                break;
            case ir::OpType::Tanh:
                if (inputs.size() == 1) return rename(eval_tanh(*inputs[0]));
                break;
            default:
                return std::nullopt;
        }
    } catch (const OptimizationError&) {
        return std::nullopt;
    }
    return std::nullopt;
}

} // namespace forge::optimizer
