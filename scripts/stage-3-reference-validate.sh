#!/usr/bin/env bash
# stage-3 — reference validation vs Python onnx. Read-only for the project.
set -u

ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/artifacts/ref-validate-$TS"
VENV="/tmp/forge-ref-venv"
mkdir -p "$OUT"/{ref,forge,compare,synth}

H() { printf '\n\033[1;33m═══ %s ═══\033[0m\n' "$*"; }

cd "$ROOT" || exit 1

# ─────────────────────────────────────────────────────────────
H "PART A — PYTHON VENV + ONNX"
# ─────────────────────────────────────────────────────────────

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" 2>&1 | tail -3
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

"$PIP" install --quiet --disable-pip-version-check onnx numpy 2>&1 | tail -5

"$PY" -c "import onnx, numpy; print(f'onnx={onnx.__version__} numpy={numpy.__version__}')" \
  | tee "$OUT/ref/versions.txt"

# ─────────────────────────────────────────────────────────────
H "PART B — REFERENCE EXTRACTION"
# ─────────────────────────────────────────────────────────────

MODEL=/root/external_test/test_model.onnx
[ -f "$MODEL" ] || { echo "FATAL: model missing"; exit 1; }
sha256sum "$MODEL" > "$OUT/ref/model.sha256"

"$PY" - "$MODEL" > "$OUT/ref/ground-truth.json" << 'PYEOF'
import sys, json
import onnx
from onnx import numpy_helper

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
        "doc_string": m.doc_string or "",
    },
    "opset": [
        {"domain": op.domain or "", "version": op.version}
        for op in m.opset_import
    ],
    "graph": {
        "name": g.name,
        "node_count": len(g.node),
        "initializer_count": len(g.initializer),
        "input_count": len(g.input),
        "output_count": len(g.output),
        "value_info_count": len(g.value_info),
    },
    "nodes": [],
    "initializers": [],
    "inputs": [],
    "outputs": [],
}

for n in g.node:
    info["nodes"].append({
        "name": n.name,
        "op_type": n.op_type,
        "domain": n.domain or "",
        "input": list(n.input),
        "output": list(n.output),
        "attribute_count": len(n.attribute),
    })

for init in g.initializer:
    elem_count = 1
    for d in init.dims:
        elem_count *= int(d)
    info["initializers"].append({
        "name": init.name,
        "dtype": int(init.data_type),
        "dims": [int(d) for d in init.dims],
        "elem_count": elem_count,
        "raw_bytes": len(init.raw_data) if init.raw_data else 0,
    })

for v in g.input:
    tt = v.type.tensor_type
    dims = [int(d.dim_value) if d.HasField('dim_value') else -1 for d in tt.shape.dim]
    info["inputs"].append({"name": v.name, "dims": dims, "dtype": int(tt.elem_type)})

for v in g.output:
    tt = v.type.tensor_type
    dims = [int(d.dim_value) if d.HasField('dim_value') else -1 for d in tt.shape.dim]
    info["outputs"].append({"name": v.name, "dims": dims, "dtype": int(tt.elem_type)})

print(json.dumps(info, indent=2, sort_keys=True))
PYEOF

echo "--- ground truth (summary) ---"
"$PY" - << PYEOF
import json
d = json.load(open("$OUT/ref/ground-truth.json"))
print(f"  ir_version       : {d['model']['ir_version']}")
print(f"  producer         : {d['model']['producer_name']} {d['model']['producer_version']}")
print(f"  opset            : {d['opset']}")
print(f"  nodes            : {d['graph']['node_count']}")
print(f"  initializers     : {d['graph']['initializer_count']}")
print(f"  inputs           : {d['graph']['input_count']}")
print(f"  outputs          : {d['graph']['output_count']}")
print(f"  value_info       : {d['graph']['value_info_count']}")
print()
print("  -- nodes --")
for n in d['nodes']:
    print(f"    {n['op_type']:14s}  in={n['input']}  out={n['output']}")
print()
print("  -- initializers --")
for i in d['initializers']:
    print(f"    {i['name']:20s}  dims={i['dims']}  elems={i['elem_count']}  bytes={i['raw_bytes']}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART C — FORGE OUTPUT (same model)"
# ─────────────────────────────────────────────────────────────

BIN=/root/forge/build-truth/forge
[ -x "$BIN" ] || BIN=/root/forge/build-test/forge
[ -x "$BIN" ] || { echo "FATAL: no forge binary"; exit 1; }

FORGE_RAW="$OUT/forge/raw.txt"
"$BIN" "$MODEL" --verify-tensors > "$FORGE_RAW" 2>&1
echo "exit: $?"

echo "--- forge raw output ---"
cat "$FORGE_RAW"

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
    "ir_version":  grab(r"IR version\s*:\s*(\d+)", raw),
    "opset":       grab(r"Opset\s*:\s*(\d+)", raw),
    "nodes":       grab(r"Nodes\s*:\s*(\d+)", raw),
    "inits":       grab(r"Initializers\s*:\s*(\d+)", raw),
    "inputs":      grab(r"Inputs\s*:\s*(\d+)", raw),
    "outputs":     grab(r"Outputs\s*:\s*(\d+)", raw),
}

ref = {
    "ir_version": str(gt["model"]["ir_version"]),
    "opset":      str(max((o["version"] for o in gt["opset"] if o["domain"]==""), default=0)),
    "nodes":      str(gt["graph"]["node_count"]),
    "inits":      str(gt["graph"]["initializer_count"]),
    "inputs":     str(gt["graph"]["input_count"]),
    "outputs":    str(gt["graph"]["output_count"]),
}

fields = ["ir_version", "opset", "nodes", "inits", "inputs", "outputs"]

print(f"{'FIELD':<14} {'REFERENCE':<12} {'FORGE':<12} {'MATCH'}")
print("-" * 50)
ok = 0
for f in fields:
    match = "OK" if ref[f] == forge[f] else "DIVERGENCE"
    if match == "OK": ok += 1
    print(f"{f:<14} {ref[f]:<12} {str(forge[f]):<12} {match}")

print()
print(f"matched: {ok}/{len(fields)}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART E — SYNTHETIC MODELS"
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

# Case 1: Relu only
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
n = helper.make_node("Relu", ["X"], ["Y"])
g = helper.make_graph([n], "relu_only", [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("relu_only.onnx", m, {"case": "relu_only", "expected_nodes": 1, "expected_inits": 0})

# Case 2: Add with constant
W = numpy_helper.from_array(np.array([1.0, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32), name="W")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
n = helper.make_node("Add", ["X", "W"], ["Y"])
g = helper.make_graph([n], "add_const", [X], [Y], [W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("add_const.onnx", m, {"case": "add_const", "expected_nodes": 1, "expected_inits": 1})

# Case 3: Small conv
W = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), name="conv_w")
B = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="conv_b")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 6, 6])
n = helper.make_node("Conv", ["X", "conv_w", "conv_b"], ["Y"], kernel_shape=[3,3])
g = helper.make_graph([n], "conv", [X], [Y], [W, B])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("conv.onnx", m, {"case": "conv", "expected_nodes": 1, "expected_inits": 2})

# Case 4: Chained ops (Conv + Relu + Add)
W1 = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), name="w1")
B1 = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="b1")
W2 = numpy_helper.from_array(np.random.randn(1, 4, 4, 4).astype(np.float32), name="w2")
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])
conv = helper.make_node("Conv", ["X", "w1", "b1"], ["conv_out"], kernel_shape=[3,3])
relu = helper.make_node("Relu", ["conv_out"], ["relu_out"])
add  = helper.make_node("Add", ["relu_out", "w2"], ["Y"])
g = helper.make_graph([conv, relu, add], "chain", [X], [Y], [W1, B1, W2])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
save("chain.onnx", m, {"case": "chain", "expected_nodes": 3, "expected_inits": 3})

# Case 5: Unsupported opset=99
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
n = helper.make_node("Relu", ["X"], ["Y"])
g = helper.make_graph([n], "future_opset", [X], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 99)])
m.ir_version = 8
save("opset99.onnx", m, {"case": "opset99", "expected_nodes": 1, "expected_inits": 0})

json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=2)
print(json.dumps(manifest, indent=2))
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART F — FORGE ON SYNTHETIC MODELS"
# ─────────────────────────────────────────────────────────────

"$PY" - << PYEOF | tee "$OUT/compare/synthetic-results.txt"
import json, subprocess, os, re

manifest = json.load(open("$OUT/synth/manifest.json"))
BIN = "$BIN"

print(f"{'CASE':<14} {'EXIT':<6} {'NODES_REF':<10} {'NODES_FORGE':<12} {'INITS_REF':<10} {'INITS_FORGE':<12} {'STATUS'}")
print("-" * 90)

for m in manifest:
    r = subprocess.run([BIN, m["path"]], capture_output=True, text=True, timeout=30)
    out = r.stdout + r.stderr
    fn = re.search(r"Nodes\s*:\s*(\d+)", out)
    fi = re.search(r"Initializers\s*:\s*(\d+)", out)
    fnv = fn.group(1) if fn else "N/A"
    fiv = fi.group(1) if fi else "N/A"

    status = "OK" if (fnv == str(m["expected_nodes"]) and fiv == str(m["expected_inits"])) else "DIVERGENCE"
    print(f"{m['case']:<14} {r.returncode:<6} {m['expected_nodes']:<10} {fnv:<12} {m['expected_inits']:<10} {fiv:<12} {status}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "MANIFEST"
# ─────────────────────────────────────────────────────────────
find "$OUT" -type f -exec sha256sum {} \; > "$OUT/manifest.sha256"
echo "artifact: $OUT"
du -sh "$OUT"
