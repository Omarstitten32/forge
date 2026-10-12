// ==========================================================
//  Coverage boost tests — fills gaps identified in
//  Phase 0.2c measurement.
// ==========================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "forge/ir/tensor.hpp"
#include "forge/ir/op.hpp"
#include "forge/ir/graph.hpp"
#include "forge/optimizer/evaluate.hpp"
#include "forge/optimizer/optimizer.hpp"
#include "forge/core/log.hpp"
#include "forge/core/error.hpp"

#include <cstring>
#include <vector>

using namespace forge;
using namespace forge::ir;
using namespace forge::optimizer;

// ==========================================================
// Helpers
// ==========================================================
static Tensor make_f32(const std::string& name,
                       std::initializer_list<int64_t> dims,
                       std::vector<float> values)
{
    TensorInfo info;
    info.name  = name;
    info.dtype = DataType::F32;
    for (auto d : dims) info.shape.dims.push_back(d);

    std::vector<std::uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return Tensor(info, std::move(bytes));
}

// ==========================================================
// dtype_name — full coverage of switch
// ==========================================================
TEST_CASE("dtype_name covers all types", "[ir][tensor]") {
    REQUIRE(std::string(dtype_name(DataType::F32))  == "f32");
    REQUIRE(std::string(dtype_name(DataType::F16))  == "f16");
    REQUIRE(std::string(dtype_name(DataType::BF16)) == "bf16");
    REQUIRE(std::string(dtype_name(DataType::I8))   == "i8");
    REQUIRE(std::string(dtype_name(DataType::I16))  == "i16");
    REQUIRE(std::string(dtype_name(DataType::I32))  == "i32");
    REQUIRE(std::string(dtype_name(DataType::I64))  == "i64");
    REQUIRE(std::string(dtype_name(DataType::U8))   == "u8");
    REQUIRE(std::string(dtype_name(DataType::Unknown)) == "unknown");
}

TEST_CASE("dtype_size covers all types", "[ir][tensor]") {
    REQUIRE(dtype_size(DataType::F32)  == 4);
    REQUIRE(dtype_size(DataType::F16)  == 2);
    REQUIRE(dtype_size(DataType::BF16) == 2);
    REQUIRE(dtype_size(DataType::I8)   == 1);
    REQUIRE(dtype_size(DataType::I16)  == 2);
    REQUIRE(dtype_size(DataType::I32)  == 4);
    REQUIRE(dtype_size(DataType::I64)  == 8);
    REQUIRE(dtype_size(DataType::U8)   == 1);
    REQUIRE(dtype_size(DataType::Unknown) == 0);
}

TEST_CASE("Tensor summary renders correctly", "[ir][tensor]") {
    auto t = make_f32("weights", {2, 3}, {1, 2, 3, 4, 5, 6});
    auto s = t.summary();

    REQUIRE(s.find("weights") != std::string::npos);
    REQUIRE(s.find("f32") != std::string::npos);
    REQUIRE(s.find("[2, 3]") != std::string::npos);
    REQUIRE(s.find("24 bytes") != std::string::npos);
}

TEST_CASE("Tensor summary without data", "[ir][tensor]") {
    TensorInfo info;
    info.name  = "empty";
    info.dtype = DataType::I32;
    info.shape = Shape({4});
    Tensor t(info);

    auto s = t.summary();
    REQUIRE(s.find("empty") != std::string::npos);
    REQUIRE(s.find("(no data)") != std::string::npos);
}

TEST_CASE("Tensor typed access", "[ir][tensor]") {
    auto t = make_f32("v", {4}, {1.5f, 2.5f, 3.5f, 4.5f});
    auto* p = t.as<float>();
    REQUIRE(p[0] == 1.5f);
    REQUIRE(p[3] == 4.5f);
}

TEST_CASE("Shape edge cases", "[ir][tensor]") {
    Shape empty;
    REQUIRE(empty.rank() == 0);
    REQUIRE(empty.total_elements() == 1);  // product of no dims
    REQUIRE(empty.to_string() == "[]");

    Shape single({5});
    REQUIRE(single.rank() == 1);
    REQUIRE(single.total_elements() == 5);

    Shape large({10, 10, 10, 10});
    REQUIRE(large.total_elements() == 10000);
}

// ==========================================================
// op_name — full coverage of switch (all ops)
// ==========================================================
TEST_CASE("op_name covers all primitive ops", "[ir][op]") {
    REQUIRE(std::string(op_name(OpType::Add))        == "Add");
    REQUIRE(std::string(op_name(OpType::Sub))        == "Sub");
    REQUIRE(std::string(op_name(OpType::Mul))        == "Mul");
    REQUIRE(std::string(op_name(OpType::Div))        == "Div");
    REQUIRE(std::string(op_name(OpType::MatMul))     == "MatMul");
    REQUIRE(std::string(op_name(OpType::Gemm))       == "Gemm");
    REQUIRE(std::string(op_name(OpType::Conv2D))     == "Conv2D");
    REQUIRE(std::string(op_name(OpType::Relu))       == "Relu");
    REQUIRE(std::string(op_name(OpType::Sigmoid))    == "Sigmoid");
    REQUIRE(std::string(op_name(OpType::Tanh))       == "Tanh");
    REQUIRE(std::string(op_name(OpType::Softmax))    == "Softmax");
    REQUIRE(std::string(op_name(OpType::Gelu))       == "Gelu");
    REQUIRE(std::string(op_name(OpType::BatchNorm))  == "BatchNorm");
    REQUIRE(std::string(op_name(OpType::LayerNorm))  == "LayerNorm");
    REQUIRE(std::string(op_name(OpType::MaxPool))    == "MaxPool");
    REQUIRE(std::string(op_name(OpType::AvgPool))    == "AvgPool");
    REQUIRE(std::string(op_name(OpType::Reshape))    == "Reshape");
    REQUIRE(std::string(op_name(OpType::Transpose))  == "Transpose");
    REQUIRE(std::string(op_name(OpType::Concat))     == "Concat");
    REQUIRE(std::string(op_name(OpType::Constant))   == "Constant");
    REQUIRE(std::string(op_name(OpType::Unknown))    == "Unknown");
}

TEST_CASE("op_name covers all fused ops", "[ir][op]") {
    REQUIRE(std::string(op_name(OpType::Conv2DRelu))         == "Conv2DRelu");
    REQUIRE(std::string(op_name(OpType::Conv2DBatchNorm))    == "Conv2DBatchNorm");
    REQUIRE(std::string(op_name(OpType::Conv2DBatchNormRelu))== "Conv2DBatchNormRelu");
    REQUIRE(std::string(op_name(OpType::AddRelu))            == "AddRelu");
    REQUIRE(std::string(op_name(OpType::MulAdd))             == "MulAdd");
}

TEST_CASE("is_fused returns correct value for all ops", "[ir][op]") {
    // Fused
    REQUIRE(is_fused(OpType::Conv2DRelu));
    REQUIRE(is_fused(OpType::Conv2DBatchNorm));
    REQUIRE(is_fused(OpType::Conv2DBatchNormRelu));
    REQUIRE(is_fused(OpType::AddRelu));
    REQUIRE(is_fused(OpType::MulAdd));

    // Not fused
    REQUIRE_FALSE(is_fused(OpType::Add));
    REQUIRE_FALSE(is_fused(OpType::Mul));
    REQUIRE_FALSE(is_fused(OpType::Conv2D));
    REQUIRE_FALSE(is_fused(OpType::Relu));
    REQUIRE_FALSE(is_fused(OpType::BatchNorm));
    REQUIRE_FALSE(is_fused(OpType::Unknown));
}

// ==========================================================
// evaluate — comprehensive
// ==========================================================
TEST_CASE("eval_sub produces correct result", "[optimizer][eval]") {
    auto a = make_f32("a", {4}, {10, 20, 30, 40});
    auto b = make_f32("b", {4}, {1, 2, 3, 4});
    auto r = eval_sub(a, b);
    auto* p = r.as<float>();
    REQUIRE(p[0] == 9.0f);
    REQUIRE(p[3] == 36.0f);
}

TEST_CASE("eval_div produces correct result", "[optimizer][eval]") {
    auto a = make_f32("a", {3}, {10, 20, 30});
    auto b = make_f32("b", {3}, {2, 4, 5});
    auto r = eval_div(a, b);
    auto* p = r.as<float>();
    REQUIRE(p[0] == 5.0f);
    REQUIRE(p[1] == 5.0f);
    REQUIRE(p[2] == 6.0f);
}

TEST_CASE("eval_sigmoid produces correct result", "[optimizer][eval]") {
    auto a = make_f32("a", {3}, {-1.0f, 0.0f, 1.0f});
    auto r = eval_sigmoid(a);
    auto* p = r.as<float>();
    // sigmoid(0) == 0.5
    REQUIRE(p[1] == Catch::Approx(0.5f).epsilon(1e-5));
    // sigmoid(1) > 0.5
    REQUIRE(p[2] > 0.5f);
    // sigmoid(-1) < 0.5
    REQUIRE(p[0] < 0.5f);
}

TEST_CASE("eval_tanh produces correct result", "[optimizer][eval]") {
    auto a = make_f32("a", {3}, {-1.0f, 0.0f, 1.0f});
    auto r = eval_tanh(a);
    auto* p = r.as<float>();
    // tanh(0) == 0
    REQUIRE(p[1] == Catch::Approx(0.0f).epsilon(1e-6));
    // |tanh(x)| < 1
    REQUIRE(std::abs(p[0]) < 1.0f);
    REQUIRE(std::abs(p[2]) < 1.0f);
    // tanh(-x) == -tanh(x)
    REQUIRE(p[0] == Catch::Approx(-p[2]).epsilon(1e-5));
}

TEST_CASE("eval_add throws on shape mismatch", "[optimizer][eval]") {
    auto a = make_f32("a", {4}, {1, 2, 3, 4});
    auto b = make_f32("b", {5}, {1, 2, 3, 4, 5});
    REQUIRE_THROWS_AS(eval_add(a, b), OptimizationError);
}

TEST_CASE("eval_add throws on non-F32", "[optimizer][eval]") {
    TensorInfo info;
    info.name  = "x";
    info.dtype = DataType::I32;
    info.shape = Shape({4});
    Tensor a(info);
    auto b = make_f32("b", {4}, {1, 2, 3, 4});
    REQUIRE_THROWS_AS(eval_add(a, b), OptimizationError);
}

// ==========================================================
// try_evaluate — covers all dispatch paths
// ==========================================================
TEST_CASE("try_evaluate handles all supported ops", "[optimizer][eval]") {
    auto a = make_f32("a", {4}, {1, 2, 3, 4});
    auto b = make_f32("b", {4}, {5, 6, 7, 8});

    TensorInfo out_info;
    out_info.name  = "out";
    out_info.dtype = DataType::F32;
    out_info.shape = Shape({4});

    Node node;
    node.id = 0;
    node.outputs = {0};

    std::vector<const Tensor*> inputs {&a, &b};

    // Add
    node.op = OpType::Add;
    REQUIRE(try_evaluate(node, inputs, out_info).has_value());

    // Sub
    node.op = OpType::Sub;
    REQUIRE(try_evaluate(node, inputs, out_info).has_value());

    // Mul
    node.op = OpType::Mul;
    REQUIRE(try_evaluate(node, inputs, out_info).has_value());

    // Div
    node.op = OpType::Div;
    REQUIRE(try_evaluate(node, inputs, out_info).has_value());

    // Unary
    std::vector<const Tensor*> unary {&a};

    node.op = OpType::Relu;
    REQUIRE(try_evaluate(node, unary, out_info).has_value());

    node.op = OpType::Sigmoid;
    REQUIRE(try_evaluate(node, unary, out_info).has_value());

    node.op = OpType::Tanh;
    REQUIRE(try_evaluate(node, unary, out_info).has_value());
}

TEST_CASE("try_evaluate returns nullopt for unsupported ops", "[optimizer][eval]") {
    auto a = make_f32("a", {4}, {1, 2, 3, 4});

    TensorInfo out_info;
    out_info.name  = "out";
    out_info.dtype = DataType::F32;
    out_info.shape = Shape({4});

    Node node;
    node.outputs = {0};

    std::vector<const Tensor*> inputs {&a};

    // Unsupported ops
    node.op = OpType::Conv2D;
    REQUIRE_FALSE(try_evaluate(node, inputs, out_info).has_value());

    node.op = OpType::MatMul;
    REQUIRE_FALSE(try_evaluate(node, inputs, out_info).has_value());

    node.op = OpType::Softmax;
    REQUIRE_FALSE(try_evaluate(node, inputs, out_info).has_value());
}

TEST_CASE("try_evaluate returns nullopt if input missing data", "[optimizer][eval]") {
    TensorInfo info;
    info.name  = "empty";
    info.dtype = DataType::F32;
    info.shape = Shape({4});
    Tensor empty(info);

    TensorInfo out_info;
    out_info.name  = "out";
    out_info.dtype = DataType::F32;
    out_info.shape = Shape({4});

    Node node;
    node.op = OpType::Relu;
    node.outputs = {0};

    std::vector<const Tensor*> inputs {&empty};
    REQUIRE_FALSE(try_evaluate(node, inputs, out_info).has_value());
}

TEST_CASE("try_evaluate returns nullopt if input null", "[optimizer][eval]") {
    TensorInfo out_info;
    out_info.name  = "out";
    out_info.dtype = DataType::F32;
    out_info.shape = Shape({4});

    Node node;
    node.op = OpType::Relu;
    node.outputs = {0};

    std::vector<const Tensor*> inputs {nullptr};
    REQUIRE_FALSE(try_evaluate(node, inputs, out_info).has_value());
}

// ==========================================================
// log.cpp — set_log_level coverage
// ==========================================================
TEST_CASE("set_log_level accepts all levels", "[core][log]") {
    REQUIRE_NOTHROW(forge::init_logging());
    REQUIRE_NOTHROW(forge::set_log_level("trace"));
    REQUIRE_NOTHROW(forge::set_log_level("debug"));
    REQUIRE_NOTHROW(forge::set_log_level("info"));
    REQUIRE_NOTHROW(forge::set_log_level("warn"));
    REQUIRE_NOTHROW(forge::set_log_level("error"));
    REQUIRE_NOTHROW(forge::set_log_level("critical"));
    REQUIRE_NOTHROW(forge::set_log_level("off"));
}

// ==========================================================
// error.hpp — all error constructors
// ==========================================================
TEST_CASE("OptimizationError contains prefix", "[core][error]") {
    OptimizationError e("bad opt");
    std::string msg = e.what();
    REQUIRE(msg.find("OptimizationError") != std::string::npos);
    REQUIRE(msg.find("bad opt") != std::string::npos);
}

TEST_CASE("CodegenError contains prefix", "[core][error]") {
    CodegenError e("bad codegen");
    std::string msg = e.what();
    REQUIRE(msg.find("CodegenError") != std::string::npos);
    REQUIRE(msg.find("bad codegen") != std::string::npos);
}

TEST_CASE("RuntimeError contains prefix", "[core][error]") {
    RuntimeError e("bad runtime");
    std::string msg = e.what();
    REQUIRE(msg.find("RuntimeError") != std::string::npos);
    REQUIRE(msg.find("bad runtime") != std::string::npos);
}

TEST_CASE("ForgeError is std::runtime_error", "[core][error]") {
    ForgeError e("base");
    const std::runtime_error& base = e;
    REQUIRE(std::string(base.what()) == "base");
}
