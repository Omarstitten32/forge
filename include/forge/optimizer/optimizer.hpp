#pragma once

#include "forge/ir/graph.hpp"
#include <cstddef>

namespace forge::optimizer {

/// Statistics from an optimization run.
struct OptimizationStats {
    std::size_t nodes_before{0};
    std::size_t nodes_after{0};
    std::size_t nodes_eliminated{0};
    std::size_t constants_folded{0};
};

/// Run all optimization passes on a graph, modifying it in place.
OptimizationStats optimize(ir::Graph& graph);

} // namespace forge::optimizer
