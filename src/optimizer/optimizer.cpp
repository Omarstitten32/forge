#include "forge/optimizer/optimizer.hpp"
#include "forge/optimizer/evaluate.hpp"
#include "forge/optimizer/fusion.hpp"
#include "forge/core/log.hpp"

#include <spdlog/spdlog.h>
#include <unordered_set>

namespace forge::optimizer {

// ==========================================================
// Pass: Constant Folding
// ==========================================================
std::size_t pass_constant_folding(ir::Graph& graph) {
    std::size_t folded = 0;
    bool changed = true;

    while (changed) {
        changed = false;
        auto topo = topological_sort(graph);

        for (auto node_id : topo) {
            auto* node = graph.node(node_id);
            if (!node) continue;
            if (node->op == ir::OpType::Constant) continue;
            if (is_fused(node->op)) continue;

            std::vector<const ir::Tensor*> inputs;
            inputs.reserve(node->inputs.size());

            bool all_const = true;
            for (auto tid : node->inputs) {
                auto* t = graph.tensor_data(tid);
                if (!t || !t->has_data()) { all_const = false; break; }
                inputs.push_back(t);
            }
            if (!all_const || inputs.empty()) continue;
            if (node->outputs.empty()) continue;

            auto out_tid = node->outputs.front();
            auto* out_info = graph.tensor(out_tid);
            if (!out_info) continue;

            auto result = try_evaluate(*node, inputs, *out_info);
            if (!result.has_value()) continue;

            *graph.tensor_data(out_tid) = std::move(*result);

            node->op = ir::OpType::Constant;
            node->inputs.clear();

            ++folded;
            changed = true;
        }
    }

    return folded;
}

// ==========================================================
// Pass: Dead Code Elimination
// ==========================================================
std::size_t pass_dead_code_elimination(ir::Graph& graph) {
    auto live = compute_live_nodes(graph);
    std::unordered_set<std::uint32_t> live_set(live.begin(), live.end());

    std::size_t removed = 0;
    for (const auto& node : graph.nodes()) {
        if (!live_set.count(node.id)) {
            graph.mark_node_dead(node.id);
            ++removed;
        }
    }
    if (removed > 0) graph.remove_dead_nodes();
    return removed;
}

// ==========================================================
// Top-level optimizer
// ==========================================================
OptimizationStats optimize(ir::Graph& graph) {
    OptimizationStats stats;
    stats.nodes_before = graph.node_count();

    spdlog::debug("[optimizer] start: {} nodes", stats.nodes_before);

    // 1. Operator fusion (creates fused ops)
    auto fusion_stats = pass_operator_fusion(graph);
    stats.operators_fused = fusion_stats.total_fusions;

    // Compact dead nodes after fusion
    if (stats.operators_fused > 0) {
        graph.remove_dead_nodes();
    }

    // 2. Constant folding
    stats.constants_folded = pass_constant_folding(graph);

    // 3. DCE
    stats.dead_nodes_removed = pass_dead_code_elimination(graph);

    stats.nodes_after = graph.node_count();
    stats.nodes_eliminated = stats.nodes_before - stats.nodes_after;

    spdlog::debug("[optimizer] done: {} nodes (folded={}, fused={}, dce={})",
                  stats.nodes_after,
                  stats.constants_folded,
                  stats.operators_fused,
                  stats.dead_nodes_removed);

    return stats;
}

} // namespace forge::optimizer
