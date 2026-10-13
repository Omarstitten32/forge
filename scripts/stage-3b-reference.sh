#!/usr/bin/env bash
# stage-3b — reference validation. No venv, direct pip.
set -u

ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/artifacts/ref-3b-$TS"
mkdir -p "$OUT"/{ref,forge,compare,synth}

H() { printf '\n\033[1;33m═══ %s ═══\033[0m\n' "$*"; }

cd "$ROOT" || exit 1

# ─────────────────────────────────────────────────────────────
H "PART A — INSTALL onnx (system pip, with --break-system-packages)"
# ─────────────────────────────────────────────────────────────

PY=python3

# 1) Check existing
if "$PY" -c "import onnx" 2>/dev/null; then
  echo "  already installed"
else
  # 2) Try apt first (Kali/Debian)
  echo "  trying apt python3-onnx ..."
  apt-get install -y --no-install-recommends python3-onnx 2>&1 | tail -3 || true

  # 3) Fallback to pip
  if ! "$PY" -c "import onnx" 2>/dev/null; then
    echo "  trying pip3 --break-system-packages ..."
    pip3 install --break-system-packages --quiet onnx numpy 2>&1 | tail -5 || true
  fi
fi

"$PY" -c "import onnx, numpy; print(f'onnx={onnx.__version__} numpy={numpy.__version__}')" \
  2>&1 | tee "$OUT/ref/versions.txt"

if ! "$PY" -c "import onnx" 2>/dev/null; then
  echo "FATAL: onnx still not importable. Reference validation impossible."
  exit 1
fi

# ─────────────────────────────────────────────────────────────
H "PART B — REFERENCE EXTRACTION (Python onnx)"
# ─────────────────────────────────────────────────────────────

MODEL=/root/external_test/test_model.onnx
[ -f "$MODEL" ] || { echo "FATAL: model missing"; exit 1; }
sha256sum "$MODEL" > "$OUT/ref/model.sha256"

"$PY" - "$MODEL" > "$OUT/ref/ground-truth.json" 2> "$OUT/ref/errors.txt" << 'PYEOF'
import sys, json
import onnx

path = sys.argv[1]
m = onnx.load(path)
g = m.graph

info = {
    "model": {
        "ir_version": m.ir_version,
        "producer_name": m.producer_name,
        "producer_version": m.producer_version,
        "domain": m.domain or "",
        "model_version": m.model_version,
    },
    "opset": [{"domain": op.domain or "", "version": op.version} for op in m.opset_import],
    "graph": {
        "name": g.name,
        "node_count": len(g.node),
        "initializer_count": len(g.initializer),
        "input_count": len(g.input),
        "output_count": len(g.output),
        "value_info_count": len(g.value_info),
    },
    "nodes": [{
        "name": n.name,
        "op_type": n.op_type,
        "domain": n.domain or "",
        "input": list(n.input),
        "output": list(n.output),
        "attribute_count": len(n.attribute),
    } for n in g.node],
    "initializers": [{
        "name": init.name,
        "dtype": int(init.data_type),
        "dims": [int(d) for d in init.dims],
        "elem_count": int(__import__('numpy').prod([int(d) for d in init.dims]) or 1),
        "raw_bytes": len(init.raw_data) if init.raw_data else 0,
    } for init in g.initializer],
    "inputs": [{
        "name": v.name,
        "dtype": int(v.type.tensor_type.elem_type),
        "dims": [int(d.dim_value) if d.HasField('dim_value') else -1 for d in v.type.tensor_type.shape.dim],
    } for v in g.input],
    "outputs": [{
        "name": v.name,
        "dtype": int(v.type.tensor_type.elem_type),
        "dims": [int(d.dim_value) if d.HasField('dim_value') else -1 for d in v.type.tensor_type.shape.dim],
    } for v in g.output],
    "value_info": [{
        "name": v.name,
        "dtype": int(v.type.tensor_type.elem_type),
        "dims": [int(d.dim_value) if d.HasField('dim_value') else -1 for d in v.type.tensor_type.shape.dim],
    } for v in g.value_info],
}
print(json.dumps(info, indent=2, sort_keys=True))
PYEOF

if [ ! -s "$OUT/ref/ground-truth.json" ]; then
  echo "FATAL: reference extraction failed:"
  cat "$OUT/ref/errors.txt"
  exit 1
fi

echo "--- ground truth ---"
"$PY" - << PYEOF
import json
d = json.load(open("$OUT/ref/ground-truth.json"))
print(f"  ir_version  : {d['model']['ir_version']}")
print(f"  producer    : {d['model']['producer_name']} {d['model']['producer_version']}")
print(f"  opset       : {d['opset']}")
print(f"  nodes       : {d['graph']['node_count']}")
print(f"  inits       : {d['graph']['initializer_count']}")
print(f"  inputs      : {d['graph']['input_count']}")
print(f"  outputs     : {d['graph']['output_count']}")
print(f"  value_info  : {d['graph']['value_info_count']}")
print()
print("  NODES:")
for n in d['nodes']:
    print(f"    {n['op_type']:14s}  in={n['input']}  out={n['output']}")
print()
print("  INITIALIZERS:")
for i in d['initializers']:
    print(f"    {i['name']:15s}  dims={i['dims']}  elems={i['elem_count']}  bytes={i['raw_bytes']}")
print()
print("  VALUE_INFO (shape annotations):")
for v in d['value_info']:
    print(f"    {v['name']:15s}  dims={v['dims']}  dtype={v['dtype']}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART C — FORGE OUTPUT (same model, --verify-tensors)"
# ─────────────────────────────────────────────────────────────

BIN=/root/forge/build-truth/forge
[ -x "$BIN" ] || BIN=/root/forge/build-test/forge
[ -x "$BIN" ] || { echo "FATAL: no forge binary"; exit 1; }

FORGE_RAW="$OUT/forge/raw.txt"
"$BIN" "$MODEL" --verify-tensors > "$FORGE_RAW" 2>&1
echo "exit: $?"

# ─────────────────────────────────────────────────────────────
H "PART D — FIELD-BY-FIELD COMPARISON"
# ─────────────────────────────────────────────────────────────

"$PY" - << PYEOF | tee "$OUT/compare/divergences.txt"
import json, re

gt = json.load(open("$OUT/ref/ground-truth.json"))
raw = open("$FORGE_RAW").read()

def grab(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else None

forge = {
    "ir_version": grab(r"IR version\s*:\s*(\d+)", raw),
    "opset":      grab(r"Opset\s*:\s*(\d+)", raw),
    "nodes":      grab(r"Nodes\s*:\s*(\d+)", raw),
    "inits":      grab(r"Initializers\s*:\s*(\d+)", raw),
    "inputs":     grab(r"Inputs\s*:\s*(\d+)", raw),
    "outputs":    grab(r"Outputs\s*:\s*(\d+)", raw),
}

ref = {
    "ir_version": str(gt["model"]["ir_version"]),
    "opset":      str(max((o["version"] for o in gt["opset"] if o["domain"]==""), default=0)),
    "nodes":      str(gt["graph"]["node_count"]),
    "inits":      str(gt["graph"]["initializer_count"]),
    "inputs":     str(gt["graph"]["input_count"]),
    "outputs":    str(gt["graph"]["output_count"]),
}

print(f"{'FIELD':<14} {'REFERENCE':<12} {'FORGE':<12} {'RESULT'}")
print("-" * 52)
ok = 0
for f in ["ir_version", "opset", "nodes", "inits", "inputs", "outputs"]:
    match = "OK" if ref[f] == forge[f] else "DIVERGENCE"
    if match == "OK": ok += 1
    print(f"{f:<14} {ref[f]:<12} {str(forge[f]):<12} {match}")
print()
print(f"matched: {ok}/6")

# Initializer deep comparison
print()
print("INITIALIZER BYTE-LEVEL COMPARISON")
print(f"{'NAME':<10} {'REF_DIMS':<14} {'REF_BYTES':<10} {'FORGE_BYTES':<12} {'RESULT'}")
print("-" * 70)

forge_tensors = {}
for line in raw.splitlines():
    m = re.match(r"TENSOR\|([^|]+)\|([^|]+)\|([^|]*)\|(\d+)\|([0-9a-f]+)", line)
    if m:
        forge_tensors[m.group(1)] = {
            "dtype": m.group(2),
            "dims": m.group(3),
            "bytes": int(m.group(4)),
            "hash": m.group(5),
        }

for i in gt["initializers"]:
    ft = forge_tensors.get(i["name"], {})
    ref_dims = ",".join(str(d) for d in i["dims"])
    fb = ft.get("bytes", "MISSING")
    result = "OK" if str(i["raw_bytes"]) == str(fb) else "DIVERGENCE"
    print(f"{i['name']:<10} {ref_dims:<14} {i['raw_bytes']:<10} {str(fb):<12} {result}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART E — SYNTHETIC MODELS (op-by-op coverage)"
# ─────────────────────────────────────────────────────────────

"$PY" - "$OUT/synth" << 'PYEOF'
import sys, os, json
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)
manifest = []

def save(name, model, meta):
    path = os.path.join(outdir, name)
    onnx.save(model, path)
    meta["path"] = path
    meta["size"] = os.path.getsize(path)
    manifest.append(meta)

# 1. Relu
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
g = helper.make_graph([helper.make_node("Relu", ["X"], ["Y"])], "relu_only", [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("01_relu.onnx", m, {"case": "relu", "ref_nodes": 1, "ref_inits": 0})

# 2. Add with constant
W = numpy_helper.from_array(np.arange(8, dtype=np.float32), name="W")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
g = helper.make_graph([helper.make_node("Add", ["X", "W"], ["Y"])], "add", [X], [Y], [W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("02_add.onnx", m, {"case": "add", "ref_nodes": 1, "ref_inits": 1})

# 3. Small Conv
W = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), name="W")
B = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="B")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 6, 6])
g = helper.make_graph([helper.make_node("Conv", ["X", "W", "B"], ["Y"], kernel_shape=[3,3])], "conv", [X], [Y], [W, B])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("03_conv.onnx", m, {"case": "conv", "ref_nodes": 1, "ref_inits": 2})

# 4. Chain
W1 = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), name="w1")
B1 = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="b1")
W2 = numpy_helper.from_array(np.random.randn(1, 4, 4, 4).astype(np.float32), name="w2")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])
g = helper.make_graph([
    helper.make_node("Conv", ["X", "w1", "b1"], ["c1"], kernel_shape=[3,3]),
    helper.make_node("Relu", ["c1"], ["r1"]),
    helper.make_node("Add", ["r1", "w2"], ["Y"]),
], "chain", [X], [Y], [W1, B1, W2])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("04_chain.onnx", m, {"case": "chain", "ref_nodes": 3, "ref_inits": 3})

# 5. Unsupported opset
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
g = helper.make_graph([helper.make_node("Relu", ["X"], ["Y"])], "future", [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 99)])
m.ir_version = 8
save("05_opset99.onnx", m, {"case": "opset99", "ref_nodes": 1, "ref_inits": 0})

# 6. Gemm
W = numpy_helper.from_array(np.random.randn(8, 4).astype(np.float32), name="W")
B = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="B")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
g = helper.make_graph([helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)], "gemm", [X], [Y], [W, B])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("06_gemm.onnx", m, {"case": "gemm", "ref_nodes": 1, "ref_inits": 2})

# 7. MaxPool
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])
g = helper.make_graph([
    helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2,2], strides=[2,2])
], "maxpool", [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("07_maxpool.onnx", m, {"case": "maxpool", "ref_nodes": 1, "ref_inits": 0})

json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=2)
print(f"generated {len(manifest)} models")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART F — FORGE ON SYNTHETIC MODELS"
# ─────────────────────────────────────────────────────────────

"$PY" - << PYEOF | tee "$OUT/compare/synthetic.txt"
import json, subprocess, re

manifest = json.load(open("$OUT/synth/manifest.json"))
BIN = "$BIN"

print(f"{'CASE':<10} {'REF_N':<7} {'FRG_N':<7} {'REF_I':<7} {'FRG_I':<7} {'EXIT':<6} {'RESULT'}")
print("-" * 70)
ok = 0
for m in manifest:
    r = subprocess.run([BIN, m["path"]], capture_output=True, text=True, timeout=30)
    out = r.stdout + r.stderr
    fn = re.search(r"Nodes\s*:\s*(\d+)", out)
    fi = re.search(r"Initializers\s*:\s*(\d+)", out)
    fnv = int(fn.group(1)) if fn else -1
    fiv = int(fi.group(1)) if fi else -1

    matched = (fnv == m["ref_nodes"] and fiv == m["ref_inits"])
    result = "OK" if matched else "DIVERGENCE"
    if matched: ok += 1
    print(f"{m['case']:<10} {m['ref_nodes']:<7} {fnv:<7} {m['ref_inits']:<7} {fiv:<7} {r.returncode:<6} {result}")
print()
print(f"matched: {ok}/{len(manifest)}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "MANIFEST"
# ─────────────────────────────────────────────────────────────

find "$OUT" -type f -exec sha256sum {} \; > "$OUT/manifest.sha256"
echo "artifact: $OUT"
du -sh "$OUT"
