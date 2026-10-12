#pragma once

#include "forge/ir/op.hpp"
#include "forge/ir/tensor.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace forge::ir {

/// A single node in the computation graph.
struct Node {
    std::uint32_t              id{0};
    OpType                     op{OpType::Unknown};
    std::string                name;
    std::vector<std::uint32_t> inputs;   // tensor IDs
    std::vector<std::uint32_t> outputs;  // tensor IDs
    std::map<std::string, AttrValue> attrs;
};

/// A computation graph.
class Graph {
public:
    Graph() = default;

    // ---- Node management ----
    std::uint32_t add_node(OpType op, std::string name = "");
    Node*         node(std::uint32_t id);
    const Node*   node(std::uint32_t id) const;
    std::size_t   node_count() const { return nodes_.size(); }

    // ---- Tensor management ----
    std::uint32_t     add_tensor(TensorInfo info);
    std::uint32_t     add_tensor(Tensor tensor);
    const TensorInfo* tensor(std::uint32_t id) const;
    Tensor*           tensor_data(std::uint32_t id);
    const Tensor*     tensor_data(std::uint32_t id) const;
    std::size_t       tensor_count() const { return tensors_.size(); }

    // Find tensor by name (returns UINT32_MAX if not found)
    std::uint32_t find_tensor(const std::string& name) const;

    // ---- Iteration ----
    const std::vector<Node>&   nodes()   const { return nodes_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }

    std::vector<Node>& nodes_mutable()   { return nodes_; }
    std::vector<Tensor>& tensors_mutable() { return tensors_; }

    // ---- Node removal (marks as dead — use remove_dead_nodes to compact) ----
    void mark_node_dead(std::uint32_t id);
    void remove_dead_nodes();

    // ---- Summary ----
    std::string summary() const;

private:
    std::vector<Node>   nodes_;
    std::vector<Tensor> tensors_;
    std::uint32_t       next_node_id_{0};
    std::uint32_t       next_tensor_id_{0};
};

} // namespace forge::ir
