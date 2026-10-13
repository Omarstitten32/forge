#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace forge::ir {

/// Operation types supported by Forge IR.
/// Fused ops are combinations that a single kernel can execute more efficiently.
enum class OpType {
    // ---- Arithmetic ----
    Add, Sub, Mul, Div,

    // ---- Matrix ----
    MatMul, Gemm,

    // ---- Convolution ----
    Conv2D,

    // ---- Activations ----
    Relu, Sigmoid, Tanh, Softmax, Gelu,

    // ---- Normalization ----
    BatchNorm, LayerNorm,

    // ---- Pooling ----
    MaxPool, AvgPool, GlobalAvgPool,

    // ---- Shape ----
    Reshape, Transpose, Concat, Flatten,

    // ---- Special ----
    Constant,
    Unknown,

    // ==========================================================
    // Fused operations (created by the fusion pass)
    // ==========================================================

    /// Conv2D followed by Relu.
    Conv2DRelu,

    /// Conv2D followed by BatchNorm.
    Conv2DBatchNorm,

    /// Conv2D + BatchNorm + Relu.
    Conv2DBatchNormRelu,

    /// Add followed by Relu.
    AddRelu,

    /// Mul followed by Add (residual scaling).
    MulAdd,
};

const char* op_name(OpType op);

/// Whether an op is a fused variant.
bool is_fused(OpType op);

/// Attribute value (variant of common types).
using AttrValue = std::variant<
    int64_t,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<double>
>;

} // namespace forge::ir
