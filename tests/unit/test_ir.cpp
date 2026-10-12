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
