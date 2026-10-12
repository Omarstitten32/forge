#pragma once

#include "forge/ir/graph.hpp"
#include <cstddef>

namespace forge::optimizer {

/// Result of a fusion pass.
struct FusionStats {
    std::size_t fused_conv_relu{0};
    std::size_t fused_conv_bn{0};
    std::size_t fused_conv_bn_relu{0};
    std::size_t fused_add_relu{0};
    std::size_t total_fusions{0};
};

/// Pattern: Conv2D followed by Relu.
/// Fusion: replace with a single Conv2DRelu node.
bool try_fuse_conv_relu(ir::Graph& g,
                        std::uint32_t conv_id,
                        std::uint32_t relu_id);

/// Pattern: Conv2D followed by BatchNorm.
bool try_fuse_conv_bn(ir::Graph& g,
                      std::uint32_t conv_id,
                      std::uint32_t bn_id);

/// Pattern: Conv2D + BatchNorm + Relu (chain).
bool try_fuse_conv_bn_relu(ir::Graph& g,
                           std::uint32_t conv_id,
                           std::uint32_t bn_id,
                           std::uint32_t relu_id);

/// Pattern: Add followed by Relu.
bool try_fuse_add_relu(ir::Graph& g,
                       std::uint32_t add_id,
                       std::uint32_t relu_id);

/// Main fusion pass — iterates until no more changes.
FusionStats pass_operator_fusion(ir::Graph& g);

} // namespace forge::optimizer
