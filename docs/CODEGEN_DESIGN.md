# Codegen Design — Phase 3

**Status:** Design
**Author:** OMAR-DEV
**Date:** 2026-10-13

---

## 1. Goal

Generate C++ code that, when compiled and run, computes the same
output as the original ONNX model given the same input.

Success criterion:

    |output_forge - output_onnx_runtime| < 1e-5

---

## 2. Scope

In Scope (Phase 3a MVP):

- Conv2D (padding, stride)
- Gemm (transB)
- Relu, Sigmoid, Tanh, Gelu
- BatchNormalization (inference)
- MaxPool, AvgPool
- Flatten, Reshape
- Add, Mul, Sub, Div

Out of Scope (Phase 3b+):

- Quantization (INT8, INT4)
- F16 / BF16
- APU dispatch
- Vulkan backend
- Dynamic shapes

---

## 3. Pipeline

    IR Graph
       |
       v
    [1] Layout Planning
    [2] Kernel Selection
    [3] Emit C++
       |
       v
    Generated .cpp + weights.bin

---

## 4. Kernels

### 4.1 Naive (reference)

Six nested loops for Conv2D. Correct but slow.

### 4.2 NEON (Phase 3b)

NEON intrinsics for 4-8x speedup.

---

## 5. Weight Embedding

Two output files:

- forge_generated.cpp : code
- forge_weights.bin   : weights

Code loads weights from file at runtime.

---

## 6. Testing

Reference: ONNX Runtime (Python).

Procedure:

1. Generate from model.onnx
2. Random input.bin
3. Compare ONNX Runtime output vs Forge output
4. Criterion: max|diff| < 1e-5

---

## 7. Milestones (Phase 3a)

- [ ] Codegen skeleton
- [ ] Conv2D naive
- [ ] Gemm naive
- [ ] Relu + activations
- [ ] BatchNorm inference
- [ ] MaxPool
- [ ] Flatten / Reshape
- [ ] Emitter
- [ ] Test on test_model
- [ ] Test on MobileNetV2

---

## 8. Success Criteria

- [ ] Compilable C++ output
- [ ] Match ONNX Runtime < 1e-5
- [ ] MobileNetV2 < 1 second

END OF DESIGN
