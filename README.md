# Forge

> Compile AI models to native ARM64 code.

**Status:** Phase 0 — Foundation

## Build
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

## Test
    cd build && ctest --output-on-failure

## Run
    ./build/forge model.onnx --backend=neon

## License
MIT (core) + Commercial (Pro SDK)
