#include "forge/optimizer/fusion.hpp"
#include "forge/optimizer/analysis.hpp"

#include <unordered_map>
#include <unordered_set>

namespace forge::optimizer {

// ==========================================================
// Helpers
// ==========================================================
namespace {

/// Count how many times a tensor is consumed by nodes.
using UseMap = std::unordered_map<std::uint32_t, std::size_t>;

/// Find the producer node of a tensor (UINT32_MAX if none).
std::uint32_t producer_of(const ir::Graph& g,
                          const std::unordered_map<std::uint32_t,
                                                   std::uint32_t>& producers,
                          std::uint32_t tid) {
    auto it = producers.find(tid);
    return (it == producers.end()) ? UINT32_MAX : it->second;
}

/// Return true if the given tensor has exactly one consumer in the graph.
bool single_consumer(const UseMap& uses, std::uint32_t tid) {
    auto it = uses.find(tid);
    return it != uses.end() && it->second == 1;
}

} // anonymous namespace

// ==========================================================
// Pattern: Conv2D + Relu
// ==========================================================
bool try_fuse_conv_relu(ir::Graph& g,
                        std::uint32_t conv_id,
                        std::uint32_t relu_id) {
    auto* conv = g.node(conv_id);
    auto* relu = g.node(relu_id);
    if (!conv || !relu) return false;
    if (conv->op != ir::OpType::Conv2D) return false;
    if (relu->op != ir::OpType::Relu)   return false;

    // Conv must have exactly 1 output; Relu must have exactly 1 input
    if (conv->outputs.size() != 1) return false;
    if (relu->inputs.size()  != 1) return false;
    if (relu->outputs.empty())     return false;

    // The single output of Conv must be the single input of Relu
    if (conv->outputs[0] != relu->inputs[0]) return false;

    // The intermediate tensor must be consumed only by Relu
    auto uses = count_tensor_uses(g);
    if (!single_consumer(uses, conv->outputs[0])) return false;

    // Perform fusion
    auto out_tensor = relu->outputs.empty() ? UINT32_MAX : relu->outputs[0];
    if (out_tensor == UINT32_MAX) return false;

    // Conv keeps its inputs (weight, input, bias)
    // Relu's output becomes Conv's output
    conv->outputs = relu->outputs;
    conv->op      = ir::OpType::Conv2DRelu;

    // Mark relu as dead — its role is now subsumed
    g.mark_node_dead(relu_id);
    return true;
}

// ==========================================================
// Pattern: Conv2D + BatchNorm
// ==========================================================
bool try_fuse_conv_bn(ir::Graph& g,
                      std::uint32_t conv_id,
                      std::uint32_t bn_id) {
    auto* conv = g.node(conv_id);
    auto* bn   = g.node(bn_id);
    if (!conv || !bn) return false;
    if (conv->op != ir::OpType::Conv2D)    return false;
    if (bn->op   != ir::OpType::BatchNorm) return false;

    if (conv->outputs.size() != 1) return false;
    if (bn->inputs.empty())        return false;
    if (bn->outputs.size() != 1)   return false;

    // BN's FIRST input must be Conv's single output.
    if (conv->outputs[0] != bn->inputs[0]) return false;

    auto uses = count_tensor_uses(g);
    if (!single_consumer(uses, conv->outputs[0])) return false;

    // Fuse: Conv keeps inputs (weight, input, bias), BN adds (scale, offset, mean, var)
    // We append BN's inputs (beyond the fused intermediate) to Conv's inputs.
    for (auto tid : bn->inputs) {
        if (tid == conv->outputs[0]) continue;  // skip the intermediate
        conv->inputs.push_back(tid);
    }
    conv->outputs = bn->outputs;
    conv->op      = ir::OpType::Conv2DBatchNorm;

    g.mark_node_dead(bn_id);
    return true;
}

// ==========================================================
// Pattern: Conv2D + BatchNorm + Relu (chain)
// ==========================================================
bool try_fuse_conv_bn_relu(ir::Graph& g,
                           std::uint32_t conv_id,
                           std::uint32_t bn_id,
                           std::uint32_t relu_id) {
    auto* conv = g.node(conv_id);
    auto* bn   = g.node(bn_id);
    auto* relu = g.node(relu_id);
    if (!conv || !bn || !relu) return false;
    if (conv->op != ir::OpType::Conv2D)    return false;
    if (bn->op   != ir::OpType::BatchNorm) return false;
    if (relu->op != ir::OpType::Relu)      return false;

    if (conv->outputs.size() != 1)  return false;
    if (bn->inputs.empty())         return false;
    if (bn->outputs.size()  != 1)   return false;
    if (relu->inputs.size() != 1)   return false;
    if (relu->outputs.size()!= 1)   return false;

    // BN's first input must be Conv's output
    if (conv->outputs[0] != bn->inputs[0])   return false;
    // Relu's only input must be BN's output
    if (bn->outputs[0]   != relu->inputs[0]) return false;

    auto uses = count_tensor_uses(g);
    if (!single_consumer(uses, conv->outputs[0])) return false;
    if (!single_consumer(uses, bn->outputs[0]))   return false;

    // Fuse all three
    for (auto tid : bn->inputs) {
        if (tid == conv->outputs[0]) continue;
        conv->inputs.push_back(tid);
    }
    conv->outputs = relu->outputs;
    conv->op      = ir::OpType::Conv2DBatchNormRelu;

    g.mark_node_dead(bn_id);
    g.mark_node_dead(relu_id);
    return true;
}

// ==========================================================
// Pattern: Add + Relu
// ==========================================================
bool try_fuse_add_relu(ir::Graph& g,
                       std::uint32_t add_id,
                       std::uint32_t relu_id) {
    auto* add  = g.node(add_id);
    auto* relu = g.node(relu_id);
    if (!add || !relu) return false;
    if (add->op  != ir::OpType::Add)  return false;
    if (relu->op != ir::OpType::Relu) return false;

    if (add->outputs.size() != 1) return false;
    if (relu->inputs.size() != 1) return false;
    if (add->outputs[0] != relu->inputs[0]) return false;

    auto uses = count_tensor_uses(g);
    if (!single_consumer(uses, add->outputs[0])) return false;

    add->outputs = relu->outputs;
    add->op      = ir::OpType::AddRelu;

    g.mark_node_dead(relu_id);
    return true;
}

// ==========================================================
// Main fusion pass
// ==========================================================
FusionStats pass_operator_fusion(ir::Graph& g) {
    FusionStats stats;
    bool changed = true;

    while (changed) {
        changed = false;

        // Build consumer map: tensor_id -> id of node that uses it as input.
        std::unordered_map<std::uint32_t, std::uint32_t> consumers;
        for (const auto& n : g.nodes()) {
            if (n.name == "__dead__") continue;
            for (auto tid : n.inputs) {
                consumers[tid] = n.id;
            }
        }

        auto uses  = count_tensor_uses(g);
        auto order = topological_sort(g);

        for (auto node_id : order) {
            auto* node = g.node(node_id);
            if (!node) continue;
            if (node->name == "__dead__") continue;

            // -------- Try Conv + BN + Relu (3-op chain) --------
            if (node->op == ir::OpType::Conv2D
                && node->outputs.size() == 1) {
                auto mid_tensor = node->outputs[0];
                if (single_consumer(uses, mid_tensor)) {
                    auto bn_id = consumers.count(mid_tensor)
                                 ? consumers[mid_tensor] : UINT32_MAX;
                    if (bn_id != UINT32_MAX) {
                        auto* bn = g.node(bn_id);
                        if (bn && bn->op == ir::OpType::BatchNorm
                            && bn->outputs.size() == 1) {
                            auto bn_out = bn->outputs[0];
                            if (single_consumer(uses, bn_out)) {
                                auto relu_id = consumers.count(bn_out)
                                               ? consumers[bn_out] : UINT32_MAX;
                                if (relu_id != UINT32_MAX) {
                                    auto* relu = g.node(relu_id);
                                    if (relu && relu->op == ir::OpType::Relu) {
                                        if (try_fuse_conv_bn_relu(
                                                g, node_id, bn_id, relu_id)) {
                                            stats.fused_conv_bn_relu++;
                                            stats.total_fusions++;
                                            changed = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // -------- Try Conv + Relu --------
            if (node->op == ir::OpType::Conv2D
                && node->outputs.size() == 1) {
                auto mid = node->outputs[0];
                if (single_consumer(uses, mid)) {
                    auto next_id = consumers.count(mid)
                                   ? consumers[mid] : UINT32_MAX;
                    if (next_id != UINT32_MAX) {
                        auto* next = g.node(next_id);
                        if (next && next->op == ir::OpType::Relu) {
                            if (try_fuse_conv_relu(g, node_id, next_id)) {
                                stats.fused_conv_relu++;
                                stats.total_fusions++;
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }

            // -------- Try Conv + BN --------
            if (node->op == ir::OpType::Conv2D
                && node->outputs.size() == 1) {
                auto mid = node->outputs[0];
                if (single_consumer(uses, mid)) {
                    auto next_id = consumers.count(mid)
                                   ? consumers[mid] : UINT32_MAX;
                    if (next_id != UINT32_MAX) {
                        auto* next = g.node(next_id);
                        if (next && next->op == ir::OpType::BatchNorm) {
                            if (try_fuse_conv_bn(g, node_id, next_id)) {
                                stats.fused_conv_bn++;
                                stats.total_fusions++;
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }

            // -------- Try Add + Relu --------
            if (node->op == ir::OpType::Add
                && node->outputs.size() == 1) {
                auto mid = node->outputs[0];
                if (single_consumer(uses, mid)) {
                    auto next_id = consumers.count(mid)
                                   ? consumers[mid] : UINT32_MAX;
                    if (next_id != UINT32_MAX) {
                        auto* next = g.node(next_id);
                        if (next && next->op == ir::OpType::Relu) {
                            if (try_fuse_add_relu(g, node_id, next_id)) {
                                stats.fused_add_relu++;
                                stats.total_fusions++;
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return stats;
}

} // namespace forge::optimizer
