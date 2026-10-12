#include <catch2/catch_test_macros.hpp>
#include "forge/ir/tensor.hpp"
#include "forge/ir/op.hpp"
#include "forge/ir/graph.hpp"

using namespace forge::ir;

TEST_CASE("dtype_size matches spec", "[ir]") {
    REQUIRE(dtype_size(DataType::F32) == 4);
    REQUIRE(dtype_size(DataType::F16) == 2);
    REQUIRE(dtype_size(DataType::I8)  == 1);
    REQUIRE(dtype_size(DataType::I32) == 4);
}

TEST_CASE("Shape arithmetic works", "[ir]") {
    Shape s({2, 3, 4});
    REQUIRE(s.rank() == 3);
    REQUIRE(s.total_elements() == 24);
    REQUIRE(s.to_string() == "[2, 3, 4]");
}

TEST_CASE("op_name is correct", "[ir]") {
    REQUIRE(std::string(op_name(OpType::Conv2D)) == "Conv2D");
    REQUIRE(std::string(op_name(OpType::Relu))   == "Relu");
}

TEST_CASE("Graph node management", "[ir]") {
    Graph g;
    auto a = g.add_node(OpType::Conv2D, "conv");
    auto b = g.add_node(OpType::Relu);

    REQUIRE(g.node_count() == 2);
    REQUIRE(g.node(a)->op == OpType::Conv2D);
    REQUIRE(g.node(a)->name == "conv");
    REQUIRE(g.node(b)->op == OpType::Relu);
}

TEST_CASE("Graph tensor management", "[ir]") {
    Graph g;
    TensorInfo t;
    t.name  = "input";
    t.dtype = DataType::F32;
    t.shape = Shape({1, 3, 224, 224});

    auto id = g.add_tensor(t);
    REQUIRE(g.tensor_count() == 1);
    REQUIRE(g.tensor(id)->name == "input");
    REQUIRE(g.tensor(id)->shape.total_elements() == 150528);
}

TEST_CASE("Tensor can hold raw data", "[ir][tensor]") {
    TensorInfo info;
    info.name  = "weights";
    info.dtype = DataType::F32;
    info.shape = Shape({2, 3});

    std::vector<std::uint8_t> data(2 * 3 * 4, 0);  // 6 floats = 24 bytes
    Tensor t(info, std::move(data));

    REQUIRE(t.has_data());
    REQUIRE(t.byte_size() == 24);
    REQUIRE(t.element_count() == 6);
    REQUIRE(t.name() == "weights");
}

TEST_CASE("Graph can store tensor with data", "[ir][graph]") {
    Graph g;
    TensorInfo info;
    info.name  = "w1";
    info.dtype = DataType::F32;
    info.shape = Shape({4});

    std::vector<std::uint8_t> data(16, 0);
    Tensor t(info, std::move(data));

    auto id = g.add_tensor(std::move(t));
    REQUIRE(g.tensor_count() == 1);
    REQUIRE(g.tensor_data(id)->has_data());
    REQUIRE(g.tensor_data(id)->byte_size() == 16);
}

TEST_CASE("Graph find_tensor by name", "[ir][graph]") {
    Graph g;
    TensorInfo info1; info1.name = "alpha";
    TensorInfo info2; info2.name = "beta";
    g.add_tensor(info1);
    g.add_tensor(info2);

    auto a = g.find_tensor("alpha");
    auto b = g.find_tensor("beta");
    auto c = g.find_tensor("gamma");

    REQUIRE(a == 0);
    REQUIRE(b == 1);
    REQUIRE(c == UINT32_MAX);
}

TEST_CASE("Graph remove_dead_nodes compacts", "[ir][graph]") {
    Graph g;
    auto id1 = g.add_node(OpType::Conv2D, "keep");
    auto id2 = g.add_node(OpType::Relu,   "remove");
    REQUIRE(g.node_count() == 2);

    g.mark_node_dead(id2);
    g.remove_dead_nodes();

    REQUIRE(g.node_count() == 1);
    REQUIRE(g.node(0)->name == "keep");
}
