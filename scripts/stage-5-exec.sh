#!/usr/bin/env bash
set -u
ROOT=/root/forge
OUT=/tmp/stage5
mkdir -p "$OUT"
cd "$ROOT"

BIN=./build-shape/forge
KERNELS=src/codegen/kernels.cpp
INCLUDE=-Iinclude

echo "═══ STEP 1 — Generate code ═══"
$BIN /root/external_test/test_model.onnx    -o "$OUT/small.cpp" 2>&1 | tail -1
$BIN /root/external_test/mobilenetv2-7.onnx -o "$OUT/mnv2.cpp"  2>&1 | tail -1

echo
echo "═══ STEP 2 — Compile kernels.o ═══"
g++ -O2 -std=c++20 $INCLUDE -c "$KERNELS" -o "$OUT/kernels.o" 2>&1 | head -5

echo
echo "═══ STEP 3 — Compile each generated + driver ═══"
for name in small mnv2; do
    SRC="$OUT/$name.cpp"
    [ -f "$SRC" ] || { echo "  $name: no source"; continue; }
    echo "  [$name]"

    cat > "$OUT/drv_$name.cpp" << 'DRVT'
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
extern "C" void forge_run(const float* input, float* output, const float* weights);

static std::vector<float> load(const char* p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(1); }
    f.seekg(0, std::ios::end);
    std::size_t n = f.tellg();
    f.seekg(0);
    std::vector<float> v(n / 4);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)n);
    return v;
}

int main(int argc, char** argv) {
    if (argc != 5) return 1;
    auto w  = load(argv[2]);
    auto in = load(argv[3]);
    std::vector<float> out(1000);
    forge_run(in.data(), out.data(), w.data());
    std::ofstream f(argv[4], std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()),
            (std::streamsize)(out.size() * 4));
    return 0;
}
DRVT

    g++ -O2 -std=c++20 $INCLUDE -c "$SRC" -o "$OUT/$name.o" 2>&1 | head -8
    if [ ! -f "$OUT/$name.o" ]; then
        echo "    → $name.o FAILED"
        continue
    fi

    g++ -O2 -std=c++20 $INCLUDE \
        "$OUT/drv_$name.cpp" "$OUT/$name.o" "$OUT/kernels.o" \
        -o "$OUT/${name}_bin" 2>&1 | head -8

    if [ -x "$OUT/${name}_bin" ]; then
        echo "    → ${name}_bin OK"
    else
        echo "    → ${name}_bin FAILED"
    fi
done

echo
echo "═══ STEP 4 — Random input ═══"
python3 - "$OUT" << 'PY'
import sys, struct, random
out = sys.argv[1]
random.seed(42)
for name, n_in in (("small", 1*3*32*32), ("mnv2", 1*3*224*224)):
    with open(f"{out}/{name}_input.bin", "wb") as f:
        f.write(struct.pack(f"{n_in}f", *[random.uniform(-1, 1) for _ in range(n_in)]))
    print(f"  {name}_input.bin: {n_in} floats")
PY

echo
echo "═══ STEP 5 — ONNX Runtime reference ═══"
python3 - "$OUT" << 'PY'
import sys, os
out = sys.argv[1]
try:
    import onnxruntime as ort
    import numpy as np
except ImportError:
    print("  installing onnxruntime ...")
    os.system("pip3 install --break-system-packages --quiet onnxruntime 2>&1 | tail -2")
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  FAILED")
        sys.exit(0)

for name, model, shape in (
    ("small", "/root/external_test/test_model.onnx",    (1,3,32,32)),
    ("mnv2",  "/root/external_test/mobilenetv2-7.onnx", (1,3,224,224)),
):
    inp = np.fromfile(f"{out}/{name}_input.bin", dtype=np.float32).reshape(shape)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    result = sess.run(None, {iname: inp})[0]
    result = np.asarray(result, dtype=np.float32).flatten()
    result.tofile(f"{out}/{name}_ort_output.bin")
    print(f"  {name}: ORT output n={result.size} sum={result.sum():.4f}")
PY

echo
echo "═══ STEP 6 — Run Forge binaries ═══"
for name in small mnv2; do
    [ -x "$OUT/${name}_bin" ] || { echo "  $name: no binary"; continue; }
    [ -f "$OUT/${name}_input.bin" ] || { echo "  $name: no input"; continue; }
    [ -f "$OUT/${name}.cpp.weights.bin" ] || { echo "  $name: no weights"; continue; }
    timeout 120 "$OUT/${name}_bin" \
        "$OUT/${name}.cpp" \
        "$OUT/${name}.cpp.weights.bin" \
        "$OUT/${name}_input.bin" \
        "$OUT/${name}_forge_output.bin"
    echo "  $name: exit=$?"
done

echo
echo "═══ STEP 7 — Compare ═══"
python3 - "$OUT" << 'PY'
import sys, os, numpy as np
out = sys.argv[1]
for name, out_n in (("small", 10), ("mnv2", 1000)):
    fp = f"{out}/{name}_forge_output.bin"
    op = f"{out}/{name}_ort_output.bin"
    if not (os.path.exists(fp) and os.path.exists(op)):
        print(f"  {name}: MISSING  forge={os.path.exists(fp)} ort={os.path.exists(op)}")
        continue
    f = np.fromfile(fp, dtype=np.float32)[:out_n]
    o = np.fromfile(op, dtype=np.float32)[:out_n]
    n = min(len(f), len(o))
    f, o = f[:n], o[:n]
    d = np.abs(f - o)
    print(f"  {name}: n={n}")
    print(f"    forge: min={f.min():+.4e} max={f.max():+.4e} sum={f.sum():+.4f}")
    print(f"    ort  : min={o.min():+.4e} max={o.max():+.4e} sum={o.sum():+.4f}")
    print(f"    max|diff|={d.max():.6e}  mean|diff|={d.mean():.6e}")
    if d.max() < 1e-4:
        print(f"    PASS (< 1e-4)")
    else:
        print(f"    FAIL (> 1e-4)")
PY
