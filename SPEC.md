# Forge — Specification

> Compile AI models to native ARM64 code.
> 10x faster, 10x smaller than ONNX Runtime on mobile.

## 1. Vision
Forge is a native AI compiler that transforms models from ONNX, GGUF, TFLite,
PyTorch into optimized ARM64 native code exploiting CPU (NEON/SVE),
MediaTek APU, and Mali GPU (Vulkan).

## 2. Problem
- No tool exploits APU 650 portably
- No tool unifies all model formats into one IR
- Mobile LLMs stuck at 1-5 tokens/sec

## 3. Pipeline
[Model] -> [Frontend] -> [IR] -> [Optimizer] -> [Codegen] -> [Runtime]

## 4. Components
- Frontend: ONNX, GGUF, TFLite, PyTorch parsers
- IR: Graph, SSA, type inference, shape propagation
- Optimizer: constant folding, fusion, quantization
- Codegen: NEON, SVE, APU, Vulkan backends
- Runtime: zero-copy, async, memory pools
- SDK: C++, C, Python, Android JNI

## 5. Targets
- Inference: 5x faster than ONNX Runtime
- Size: 10x smaller (200 MB -> 20 MB)
- Memory: 3x less
- APU utilization: 80%+

## 6. Business
Community (free), Pro SDK ($499/yr), Team ($2,999/yr), Enterprise ($20K+/yr)

## 7. Roadmap
Phase 0-8 over 8-10 months

## 8. License
MIT core + Commercial Pro SDK

Version: 0.1.0-draft
Author: OMAR-DEV
