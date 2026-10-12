#pragma once

#include "forge/ir/graph.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace forge::optimizer {

/// Result of a whole-graph analysis.
struct GraphAnalysis {
    /// How many times each tensor is used as an input by nodes.
    /// tensor_id -> use_count
    std::unordered_map<std::uint32_t, std::size_t> tensor_use_count;

    /// Which node produced each tensor (UINT32_MAX if input/initializer).
    /// tensor_id -> producer node id
    std::unordered_map<std::uint32_t, std::uint32_t> tensor_producer;

    /// Topologically sorted node IDs.
    std::vector<std::uint32_t> topo_order;

    /// Nodes reachable from graph outputs (backward slice).
    std::vector<std::uint32_t> live_nodes;

    /// Number of dead (unreachable) nodes.
    std::size_t dead_node_count{0};
};

/// Compute full graph analysis. O(V + E).
GraphAnalysis analyze(const ir::Graph& g);

/// Build topological order. Throws OptimizationError on cycle.
std::vector<std::uint32_t> topological_sort(const ir::Graph& g);

/// Count tensor uses (as node inputs).
std::unordered_map<std::uint32_t, std::size_t>
count_tensor_uses(const ir::Graph& g);

/// Map tensors to their producer node.
std::unordered_map<std::uint32_t, std::uint32_t>
map_tensor_producers(const ir::Graph& g);

/// Compute backward slice: which nodes contribute to graph outputs.
std::vector<std::uint32_t>
compute_live_nodes(const ir::Graph& g);

} // namespace forge::optimizer
