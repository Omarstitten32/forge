#include "forge/ir/graph.hpp"
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
    tensors_.push_back(std::move(info));
    return next_tensor_id_++;
}

const TensorInfo* Graph::tensor(std::uint32_t id) const {
    if (id >= tensors_.size()) return nullptr;
    return &tensors_[id];
}

std::string Graph::summary() const {
    std::ostringstream os;
    os << "Graph: " << nodes_.size() << " nodes, "
       << tensors_.size() << " tensors";
    return os.str();
}

} // namespace forge::ir
