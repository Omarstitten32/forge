#pragma once

#include "forge/ir/graph.hpp"
#include "forge/ir/tensor.hpp"
#include <cstdint>
#include <optional>

namespace forge::optimizer {

/// Low-level evaluation primitives.
/// All functions assume tensor data exists and dtypes match.

/// Element-wise binary operations on F32 tensors.
/// Both must have identical shape and element count.
ir::Tensor eval_add(const ir::Tensor& a, const ir::Tensor& b);
ir::Tensor eval_sub(const ir::Tensor& a, const ir::Tensor& b);
ir::Tensor eval_mul(const ir::Tensor& a, const ir::Tensor& b);
ir::Tensor eval_div(const ir::Tensor& a, const ir::Tensor& b);

/// Element-wise unary operations.
ir::Tensor eval_relu(const ir::Tensor& a);
ir::Tensor eval_sigmoid(const ir::Tensor& a);
ir::Tensor eval_tanh(const ir::Tensor& a);

/// Try to evaluate a node given its input tensors.
/// Returns std::nullopt if the operation is unsupported
/// or if any input is missing data.
std::optional<ir::Tensor> try_evaluate(
    const ir::Node& node,
    const std::vector<const ir::Tensor*>& inputs,
    const ir::TensorInfo& output_info
);

} // namespace forge::optimizer
