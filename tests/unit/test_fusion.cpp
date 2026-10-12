#include <catch2/catch_test_macros.hpp>
#include "forge/ir/graph.hpp"
#include "forge/optimizer/fusion.hpp"
#include "forge/optimizer/optimizer.hpp"

using namespace forge;
using namespace forge::ir;
using namespace forge::optimizer;

// ==========================================================
// Helpers
// ==========================================================
static std::uint32_t add_tensor_anon(Graph& g, const std::string& name) {
    TensorInfo info;
    info.name = name;
    info.dtype = DataType::F32;
    info.shape = Shape({1, 3, 32, 32});
    return g.add_tensor(info);
}

// ==========================================================
// Conv + Relu
// ==========================================================
TEST_CASE("Fuse Conv2D + Relu", "[optimizer][fusion]") {
    Graph g;

    auto t_in  = add_tensor_anon(g, "input");
    auto t_w   = add_tensor_anon(g, "weight");
    auto t_b   = add_tensor_anon(g, "bias");
    auto t_mid = add_tensor_anon(g, "mid");
    auto t_out = add_tensor_anon(g, "out");

    auto conv = g.add_node(OpType::Conv2D, "conv");
    g.node(conv)->inputs  = {t_in, t_w, t_b};
    g.node(conv)->outputs = {t_mid};

    auto relu = g.add_node(OpType::Relu, "relu");
    g.node(relu)->inputs  = {t_mid};
    g.node(relu)->outputs = {t_out};

    auto stats = pass_operator_fusion(g);
    REQUIRE(stats.fused_conv_relu == 1);
    REQUIRE(stats.total_fusions == 1);

    g.remove_dead_nodes();
    REQUIRE(g.node_count() == 1);
    REQUIRE(g.node(0)->op == OpType::Conv2DRelu);
}

TEST_CASE("Conv without Relu is not fused", "[optimizer][fusion]") {
    Graph g;
    auto t_in  = add_tensor_anon(g, "in");
    auto t_w   = add_tensor_anon(g, "w");
    auto t_b   = add_tensor_anon(g, "b");
    auto t_out = add_tensor_anon(g, "out");

    auto conv = g.add_node(OpType::Conv2D, "conv");
    g.node(conv)->inputs  = {t_in, t_w, t_b};
    g.node(conv)->outputs = {t_out};

    auto stats = pass_operator_fusion(g);
    REQUIRE(stats.total_fusions == 0);
}

// ==========================================================
// Add + Relu
// ==========================================================
TEST_CASE("Fuse Add + Relu", "[optimizer][fusion]") {
    Graph g;

    auto t_a   = add_tensor_anon(g, "a");
    auto t_b   = add_tensor_anon(g, "b");
    auto t_mid = add_tensor_anon(g, "mid");
    auto t_out = add_tensor_anon(g, "out");

    auto add = g.add_node(OpType::Add, "add");
    g.node(add)->inputs  = {t_a, t_b};
    g.node(add)->outputs = {t_mid};

    auto relu = g.add_node(OpType::Relu, "relu");
    g.node(relu)->inputs  = {t_mid};
    g.node(relu)->outputs = {t_out};

    auto stats = pass_operator_fusion(g);
    REQUIRE(stats.fused_add_relu == 1);

    g.remove_dead_nodes();
    REQUIRE(g.node_count() == 1);
    REQUIRE(g.node(0)->op == OpType::AddRelu);
}

// ==========================================================
// Conv + BatchNorm + Relu (chain)
// ==========================================================
TEST_CASE("Fuse Conv2D + BatchNorm + Relu", "[optimizer][fusion]") {
    Graph g;
    auto t_in   = add_tensor_anon(g, "in");
    auto t_w    = add_tensor_anon(g, "w");
    auto t_b    = add_tensor_anon(g, "b");
    auto t_mid1 = add_tensor_anon(g, "mid1");
    auto t_mid2 = add_tensor_anon(g, "mid2");
    auto t_scale= add_tensor_anon(g, "scale");
    auto t_off  = add_tensor_anon(g, "offset");
    auto t_mean = add_tensor_anon(g, "mean");
    auto t_var  = add_tensor_anon(g, "var");
    auto t_out  = add_tensor_anon(g, "out");

    auto conv = g.add_node(OpType::Conv2D, "conv");
    g.node(conv)->inputs  = {t_in, t_w, t_b};
    g.node(conv)->outputs = {t_mid1};

    auto bn = g.add_node(OpType::BatchNorm, "bn");
    g.node(bn)->inputs  = {t_mid1, t_scale, t_off, t_mean, t_var};
    g.node(bn)->outputs = {t_mid2};

    auto relu = g.add_node(OpType::Relu, "relu");
    g.node(relu)->inputs  = {t_mid2};
    g.node(relu)->outputs = {t_out};

    auto stats = pass_operator_fusion(g);
    REQUIRE(stats.fused_conv_bn_relu == 1);

    g.remove_dead_nodes();
    REQUIRE(g.node_count() == 1);
    REQUIRE(g.node(0)->op == OpType::Conv2DBatchNormRelu);
}

// ==========================================================
// is_fused helper
// ==========================================================
TEST_CASE("is_fused identifies fused ops", "[ir][op]") {
    REQUIRE(is_fused(OpType::Conv2DRelu));
    REQUIRE(is_fused(OpType::Conv2DBatchNorm));
    REQUIRE(is_fused(OpType::Conv2DBatchNormRelu));
    REQUIRE(is_fused(OpType::AddRelu));
    REQUIRE(is_fused(OpType::MulAdd));
    REQUIRE_FALSE(is_fused(OpType::Conv2D));
    REQUIRE_FALSE(is_fused(OpType::Relu));
    REQUIRE_FALSE(is_fused(OpType::Add));
}
