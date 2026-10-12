"""
Create a small ONNX model for testing Forge.
Model: Conv2D -> Relu -> MaxPool -> Flatten -> Gemm
"""
import onnx
from onnx import helper, TensorProto

# Input/Output
input_tensor = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 3, 32, 32]
)
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 10]
)

# Weights
conv_w = helper.make_tensor(
    name="conv_w", data_type=TensorProto.FLOAT,
    dims=[8, 3, 3, 3], vals=[0.01] * (8 * 3 * 3 * 3),
)
conv_b = helper.make_tensor(
    name="conv_b", data_type=TensorProto.FLOAT,
    dims=[8], vals=[0.0] * 8,
)
gemm_w = helper.make_tensor(
    name="gemm_w", data_type=TensorProto.FLOAT,
    dims=[1800, 10], vals=[0.001] * (1800 * 10),
)
gemm_b = helper.make_tensor(
    name="gemm_b", data_type=TensorProto.FLOAT,
    dims=[10], vals=[0.0] * 10,
)

# Nodes
nodes = [
    helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv_out"],
                     name="conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu1"),
    helper.make_node("MaxPool", ["relu_out"], ["pool_out"],
                     name="pool1", kernel_shape=[2, 2], strides=[2, 2]),
    helper.make_node("Flatten", ["pool_out"], ["flat_out"],
                     name="flatten1", axis=1),
    helper.make_node("Gemm", ["flat_out", "gemm_w", "gemm_b"], ["output"],
                     name="gemm1", alpha=1.0, beta=1.0, transB=0),
]

# Graph
graph = helper.make_graph(
    nodes=nodes, name="TestMLP",
    inputs=[input_tensor], outputs=[output_tensor],
    initializer=[conv_w, conv_b, gemm_w, gemm_b],
)

# Model
model = helper.make_model(
    graph, producer_name="forge-test", producer_version="0.1.0",
    opset_imports=[helper.make_opsetid("", 17)],
)
model.ir_version = 10

# Save
onnx.save(model, "test_model.onnx")

# Verify
loaded = onnx.load("test_model.onnx")
onnx.checker.check_model(loaded)

print("✅ Model created: test_model.onnx")
print(f"   IR version   : {loaded.ir_version}")
print(f"   Producer     : {loaded.producer_name} {loaded.producer_version}")
print(f"   Nodes        : {len(loaded.graph.node)}")
print(f"   Initializers : {len(loaded.graph.initializer)}")
print(f"   Inputs       : {len(loaded.graph.input)}")
print(f"   Outputs      : {len(loaded.graph.output)}")
