#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace forge::ir {

/// Operation types supported by Forge IR.
enum class OpType {
    // Arithmetic
    Add, Sub, Mul, Div,
    // Matrix
    MatMul, Gemm,
    // Convolution
    Conv2D,
    // Activations
    Relu, Sigmoid, Tanh, Softmax, Gelu,
    // Normalization
    BatchNorm, LayerNorm,
    // Pooling
    MaxPool, AvgPool,
    // Shape
    Reshape, Transpose, Concat,
    // Special
    Constant,
    Unknown,
};

const char* op_name(OpType op);

/// Attribute value (variant of common types).
using AttrValue = std::variant<
    int64_t,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<double>
>;

} // namespace forge::ir
