#include "forge/ir/graph.hpp"
#include <algorithm>
#include <sstream>

namespace forge::ir {

std::uint32_t Graph::add_node(OpType op, std::string name) {
    Node n;
    n.id   = next_node_id_++;
    n.op   = op;
    n.name = name.empty() ? std::string(op_name(op)) : std::move(name);
    nodes_.push_back(std::move(n));
    return nodes_.back().id;
}

Node* Graph::node(std::uint32_t id) {
    for (auto& n : nodes_) if (n.id == id) return &n;
    return nullptr;
}

const Node* Graph::node(std::uint32_t id) const {
    for (const auto& n : nodes_) if (n.id == id) return &n;
    return nullptr;
}

std::uint32_t Graph::add_tensor(TensorInfo info) {
    auto id = next_tensor_id_++;
    tensors_.emplace_back(std::move(info));
    return id;
}

std::uint32_t Graph::add_tensor(Tensor tensor) {
    auto id = next_tensor_id_++;
    tensors_.push_back(std::move(tensor));
    return id;
}

const TensorInfo* Graph::tensor(std::uint32_t id) const {
    if (id >= tensors_.size()) return nullptr;
    return &tensors_[id].info();
}

Tensor* Graph::tensor_data(std::uint32_t id) {
    if (id >= tensors_.size()) return nullptr;
    return &tensors_[id];
}

const Tensor* Graph::tensor_data(std::uint32_t id) const {
    if (id >= tensors_.size()) return nullptr;
    return &tensors_[id];
}

std::uint32_t Graph::find_tensor(const std::string& name) const {
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        if (tensors_[i].name() == name) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return UINT32_MAX;
}

void Graph::mark_node_dead(std::uint32_t id) {
    for (auto& n : nodes_) {
        if (n.id == id) {
            n.op   = OpType::Unknown;
            n.name = "__dead__";
            n.inputs.clear();
            n.outputs.clear();
            return;
        }
    }
}

void Graph::remove_dead_nodes() {
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [](const Node& n) { return n.name == "__dead__"; }),
        nodes_.end()
    );
}

std::string Graph::summary() const {
    std::ostringstream os;
    std::size_t with_data = 0;
    for (const auto& t : tensors_) {
        if (t.has_data()) ++with_data;
    }
    os << "Graph: " << nodes_.size() << " nodes, "
       << tensors_.size() << " tensors ("
       << with_data << " with data)";
    return os.str();
}

} // namespace forge::ir
