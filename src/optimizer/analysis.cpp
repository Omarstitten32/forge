#include "forge/optimizer/analysis.hpp"
#include "forge/core/error.hpp"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace forge::optimizer {

// ==========================================================
// count_tensor_uses
// ==========================================================
std::unordered_map<std::uint32_t, std::size_t>
count_tensor_uses(const ir::Graph& g) {
    std::unordered_map<std::uint32_t, std::size_t> counts;
    for (const auto& node : g.nodes()) {
        for (auto tid : node.inputs) {
            counts[tid]++;
        }
    }
    return counts;
}

// ==========================================================
// map_tensor_producers
// ==========================================================
std::unordered_map<std::uint32_t, std::uint32_t>
map_tensor_producers(const ir::Graph& g) {
    std::unordered_map<std::uint32_t, std::uint32_t> producers;
    for (const auto& node : g.nodes()) {
        for (auto tid : node.outputs) {
            producers[tid] = node.id;
        }
    }
    return producers;
}

// ==========================================================
// topological_sort (Kahn's algorithm)
// ==========================================================
std::vector<std::uint32_t> topological_sort(const ir::Graph& g) {
    const auto& nodes = g.nodes();
    const std::size_t n = nodes.size();

    // Build in-degree map: how many distinct producers each node waits for.
    std::unordered_map<std::uint32_t, std::size_t> in_degree;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> producers_of;
    // producer_of[tensor] -> list of node ids waiting on it.

    const auto prod_map = map_tensor_producers(g);

    for (const auto& node : nodes) {
        in_degree[node.id] = 0;
    }

    for (const auto& node : nodes) {
        for (auto in_tid : node.inputs) {
            auto it = prod_map.find(in_tid);
            if (it != prod_map.end()) {
                producers_of[it->second].push_back(node.id);
                in_degree[node.id]++;
            }
        }
    }

    // Queue of nodes with in-degree 0
    std::queue<std::uint32_t> ready;
    for (const auto& node : nodes) {
        if (in_degree[node.id] == 0) {
            ready.push(node.id);
        }
    }

    std::vector<std::uint32_t> order;
    order.reserve(n);

    while (!ready.empty()) {
        auto id = ready.front();
        ready.pop();
        order.push_back(id);

        for (auto consumer : producers_of[id]) {
            if (--in_degree[consumer] == 0) {
                ready.push(consumer);
            }
        }
    }

    if (order.size() != n) {
        throw OptimizationError(
            "Cycle detected in graph (topological sort incomplete)");
    }

    return order;
}

// ==========================================================
// compute_live_nodes (backward slice from outputs)
// ==========================================================
std::vector<std::uint32_t> compute_live_nodes(const ir::Graph& g) {
    // A node is "live" if it is a producer of at least one graph output
    // (or a tensor that flows into a live node).

    // We identify graph outputs as tensors with use_count == 0
    // (nobody consumes them) but which are outputs of some node.
    const auto uses = count_tensor_uses(g);
    const auto prod_map = map_tensor_producers(g);

    // Start from "terminal" nodes — nodes whose outputs are not consumed
    // by any other node. These are effectively graph outputs.
    std::unordered_set<std::uint32_t> live;
    std::queue<std::uint32_t> q;

    auto producer_of_node = [&](const ir::Node& node) {
        // No-op helper for clarity; not used
        (void)node;
    };

    // Seed: nodes with at least one output tensor that has 0 uses
    for (const auto& node : g.nodes()) {
        for (auto out_tid : node.outputs) {
            auto it = uses.find(out_tid);
            std::size_t uc = (it == uses.end()) ? 0 : it->second;
            if (uc == 0) {
                if (live.insert(node.id).second) {
                    q.push(node.id);
                }
                break;
            }
        }
    }

    (void)producer_of_node;

    // Propagate backward through inputs.
    while (!q.empty()) {
        auto nid = q.front(); q.pop();
        const auto* node = g.node(nid);
        if (!node) continue;

        for (auto in_tid : node->inputs) {
            auto it = prod_map.find(in_tid);
            if (it == prod_map.end()) continue;
            auto producer_node_id = it->second;
            if (live.insert(producer_node_id).second) {
                q.push(producer_node_id);
            }
        }
    }

    // Return in topological order for determinism.
    auto topo = topological_sort(g);
    std::vector<std::uint32_t> result;
    result.reserve(live.size());
    for (auto id : topo) {
        if (live.count(id)) result.push_back(id);
    }
    return result;
}

// ==========================================================
// analyze — full graph analysis
// ==========================================================
GraphAnalysis analyze(const ir::Graph& g) {
    GraphAnalysis a;
    a.tensor_use_count = count_tensor_uses(g);
    a.tensor_producer  = map_tensor_producers(g);
    a.topo_order       = topological_sort(g);
    a.live_nodes       = compute_live_nodes(g);
    a.dead_node_count  = g.node_count() - a.live_nodes.size();
    return a;
}

} // namespace forge::optimizer
