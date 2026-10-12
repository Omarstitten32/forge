# Forge

> Compile AI models to native ARM64 code.
> Fast. Small. Portable.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20android-blue.svg)]()
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

---

## What is Forge?

**Forge** is a native AI compiler that transforms models from ONNX
(and later GGUF, TFLite, PyTorch) into optimized ARM64 code that runs
directly on mobile devices, exploiting CPU SIMD (NEON/SVE), MediaTek APU,
and Mali GPU (Vulkan).

No Python runtime. No cloud. No ONNX Runtime.

---

## Status - Phase 2 (Alpha)

### What works today

- **ONNX Parser** - reads any ONNX model (Opset 7 through 17+)
- **IR** - unified graph with nodes, tensors, attributes
- **Optimizer** - 3 passes:
  - **Operator Fusion** - 5 patterns
  - **Constant Folding** - element-wise evaluation
  - **Dead Code Elimination** - removes unreachable nodes
- **CLI** - reads models, shows analysis, applies optimizations

### Validated on MobileNetV2

| Metric | Value |
|-------|-------|
| Model size | 14 MB |
| Initial nodes | 155 |
| Initializers | 267 |
| **Nodes after optimization** | **66** |
| **Reduction** | **57%** |
| Processing time | **88 ms** |
| Weights loaded | 13.5 MB |

### Test suite

- **79 assertions** across **25 test cases**
-

---

## Quick Start

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/forge-tests
./build/forge model.onnx
```

### Example Output

```
[info] Forge v0.1.0
[info] Input:   mobilenetv2-7.onnx
[info] Backend: neon
[info]
[info] ===== ONNX Metadata =====
[info]   IRversion : 3
[info]   Opset      : 7
[info]   Nodes      : 155
[info]   Initializers: 267
[info]
[info] ===== IR Graph =====
[info]   Graph: 155 nodes, 423 tensors (267 with data)
[info]
[info] Optimization: 155 -> 66 nodes
```

---

## Architecture

```
[Model File] -> [Frontend] -> [IR] -> [Optimizer] -> [Codegen] -> [Runtime]
```

### Modules

| Module | Description | LOC |
|--------|-------------|-----|
| forge-core | Logging, errors | 90 |
| forge-ir | Graph, tensors, ops | 444 |
| forge-frontend | ONNX parser | 373 |
| forge-optimizer | Fusion, folding, DCE | 903 |
| forge-codegen | Backend generators | 48 |
| forge-runtime | Execution context | 43 |
| forge-cli | CLI interface | 94 |
| Tests | Catch2 suite | 425 |

**Total: ~2,400 LOC of C++20**

---

## Roadmap

- [x] Phase 0 - Foundation
- [x] Phase 1 - ONNX parser
- [x] Phase 2 - Optimizer
- [ ] Phase 3 - NEON codegen
-
[ ] Phase 4 - APU integration
- [ ] Phase 5 - Runtime engine
- [ ] Phase 6 - Multi-format
- [ ] Phase 7 - Vulkan backend
- [ ] Phase 8 - Public beta

---

## Why Forge

| Feature | llama.cpp | ONNX Runtime | TFLite | Forge |
|---------|---------|-------------|-------|-------|
| Size | 5 MB | 200 MB | 3 MB | 616 KB |
| ONNX support | No | Yes | No | Yes |
| APU support | No | No | Partial | Planned |
| Multi-format | No | No | No | Planned |
| Vulkan | No | No | No | Planned |
| Zero deps | Yes | No | No | Yes |

---

## License

[MIT](LICENSE) (c) 2026 OMAR-DEV (Omarstitten32)
