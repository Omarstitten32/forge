#pragma once

#include "forge/ir/graph.hpp"
#include "forge/optimizer/analysis.hpp"
#include <cstddef>

namespace forge::optimizer {

struct OptimizationStats {
    std::size_t nodes_before{0};
    std::size_t nodes_after{0};
    std::size_t nodes_eliminated{0};
    std::size_t constants_folded{0};
    std::size_t dead_nodes_removed{0};
};

/// Run all optimization passes in order.
OptimizationStats optimize(ir::Graph& graph);

/// Pass 1: Constant Folding.
/// Replaces nodes whose inputs are all constants with a single constant tensor.
std::size_t pass_constant_folding(ir::Graph& graph);

/// Pass 2: Dead Code Elimination.
std::size_t pass_dead_code_elimination(ir::Graph& graph);

} // namespace forge::optimizer
