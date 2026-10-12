#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "forge/ir/graph.hpp"
#include "forge/optimizer/optimizer.hpp"
#include "forge/optimizer/evaluate.hpp"
#include "forge/optimizer/analysis.hpp"

using namespace forge;
using namespace forge::ir;

// Helper: create an F32 tensor with given values
static Tensor make_f32(const std::string& name,
                       const Shape& shape,
                       const std::vector<float>& values) {
    TensorInfo info;
    info.name  = name;
    info.dtype = DataType::F32;
    info.shape = shape;

    std::vector<std::uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return Tensor(info, std::move(bytes));
}

// ==========================================================
// Element-wise evaluation
// ==========================================================
TEST_CASE("eval_add produces correct result", "[optimizer][eval]") {
    auto a = make_f32("a", Shape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_f32("b", Shape({4}), {10.0f, 20.0f, 30.0f, 40.0f});

    auto r = optimizer::eval_add(a, b);
    REQUIRE(r.has_data());
    REQUIRE(r.element_count() == 4);

    const float* p = r.as<float>();
    REQUIRE(p[0] == 11.0f);
    REQUIRE(p[1] == 22.0f);
    REQUIRE(p[2] == 33.0f);
    REQUIRE(p[3] == 44.0f);
}

TEST_CASE("eval_mul produces correct result", "[optimizer][eval]") {
    auto a = make_f32("a", Shape({3}), {2.0f, 3.0f, 4.0f});
    auto b = make_f32("b", Shape({3}), {5.0f, 6.0f, 7.0f});

    auto r = optimizer::eval_mul(a, b);
    const float* p = r.as<float>();
    REQUIRE(p[0] == 10.0f);
    REQUIRE(p[1] == 18.0f);
    REQUIRE(p[2] == 28.0f);
}

TEST_CASE("eval_relu clamps negatives", "[optimizer][eval]") {
    auto a = make_f32("a", Shape({4}), {-1.0f, 0.0f, 1.0f, -3.0f});
    auto r = optimizer::eval_relu(a);
    const float* p = r.as<float>();
    REQUIRE(p[0] == 0.0f);
    REQUIRE(p[1] == 0.0f);
    REQUIRE(p[2] == 1.0f);
    REQUIRE(p[3] == 0.0f);
}

// ==========================================================
// Analysis
// ==========================================================
TEST_CASE("count_tensor_uses counts inputs", "[optimizer][analysis]") {
    Graph g;
    auto t0 = g.add_tensor(TensorInfo{"t0", DataType::F32, Shape({4})});
    auto t1 = g.add_tensor(TensorInfo{"t1", DataType::F32, Shape({4})});
    auto t2 = g.add_tensor(TensorInfo{"t2", DataType::F32, Shape({4})});

    auto n0 = g.add_node(OpType::Add, "n0");
    g.node(n0)->inputs = {t0, t1};
    g.node(n0)->outputs = {t2};

    auto uses = optimizer::count_tensor_uses(g);
    REQUIRE(uses[t0] == 1);
    REQUIRE(uses[t1] == 1);
    REQUIRE(uses.count(t2) == 0);   // t2 is not used as input
}

TEST_CASE("topological_sort orders nodes", "[optimizer][analysis]") {
    Graph g;
    auto t_in  = g.add_tensor(TensorInfo{"in",  DataType::F32, Shape({4})});
    auto t_mid = g.add_tensor(TensorInfo{"mid", DataType::F32, Shape({4})});
    auto t_out = g.add_tensor(TensorInfo{"out", DataType::F32, Shape({4})});

    auto n0 = g.add_node(OpType::Add, "n0");
    g.node(n0)->inputs  = {t_in};
    g.node(n0)->outputs = {t_mid};

    auto n1 = g.add_node(OpType::Relu, "n1");
    g.node(n1)->inputs  = {t_mid};
    g.node(n1)->outputs = {t_out};

    auto topo = optimizer::topological_sort(g);
    REQUIRE(topo.size() == 2);
    REQUIRE(topo[0] == n0);
    REQUIRE(topo[1] == n1);
}

// ==========================================================
// Constant Folding
// ==========================================================
TEST_CASE("constant folding computes Add", "[optimizer][fold]") {
    Graph g;
    auto t_a = g.add_tensor(make_f32("a", Shape({4}), {1, 2, 3, 4}));
    auto t_b = g.add_tensor(make_f32("b", Shape({4}), {10, 20, 30, 40}));
    auto t_c = g.add_tensor(TensorInfo{"c", DataType::F32, Shape({4})});

    auto nid = g.add_node(OpType::Add, "add");
    g.node(nid)->inputs  = {t_a, t_b};
    g.node(nid)->outputs = {t_c};

    auto folded = optimizer::pass_constant_folding(g);
    REQUIRE(folded == 1);

    auto* result = g.tensor_data(t_c);
    REQUIRE(result->has_data());
    const float* p = result->as<float>();
    REQUIRE(p[0] == 11.0f);
    REQUIRE(p[3] == 44.0f);
}

TEST_CASE("constant folding leaves non-const untouched", "[optimizer][fold]") {
    Graph g;
    auto t_a = g.add_tensor(make_f32("a", Shape({4}), {1, 2, 3, 4}));
    auto t_b = g.add_tensor(TensorInfo{"b", DataType::F32, Shape({4})});  // no data
    auto t_c = g.add_tensor(TensorInfo{"c", DataType::F32, Shape({4})});

    auto nid = g.add_node(OpType::Add, "add");
    g.node(nid)->inputs  = {t_a, t_b};
    g.node(nid)->outputs = {t_c};

    auto folded = optimizer::pass_constant_folding(g);
    REQUIRE(folded == 0);
    REQUIRE_FALSE(g.tensor_data(t_c)->has_data());
}

// ==========================================================
// Full pipeline
// ==========================================================
TEST_CASE("full optimize pipeline", "[optimizer]") {
    Graph g;
    auto t_a = g.add_tensor(make_f32("a", Shape({4}), {1, 2, 3, 4}));
    auto t_b = g.add_tensor(make_f32("b", Shape({4}), {10, 20, 30, 40}));
    auto t_c = g.add_tensor(TensorInfo{"c", DataType::F32, Shape({4})});

    auto nid = g.add_node(OpType::Add, "add");
    g.node(nid)->inputs  = {t_a, t_b};
    g.node(nid)->outputs = {t_c};

    auto stats = optimizer::optimize(g);
    REQUIRE(stats.constants_folded == 1);
    REQUIRE(g.tensor_data(t_c)->has_data());
}
