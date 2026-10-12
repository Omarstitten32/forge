#include "forge/optimizer/optimizer.hpp"

namespace forge::optimizer {

OptimizationStats optimize(ir::Graph& graph) {
    OptimizationStats stats;
    stats.nodes_before = graph.node_count();

    // TODO(Phase 2): constant folding, fusion, DCE, quantization

    stats.nodes_after = graph.node_count();
    stats.nodes_eliminated = stats.nodes_before - stats.nodes_after;
    return stats;
}

} // namespace forge::optimizer
