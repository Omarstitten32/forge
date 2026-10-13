#!/usr/bin/env bash
# stage-3c — MobileNetV2 reference validation. Raw facts only.
set -u

ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/artifacts/ref-3c-$TS"
mkdir -p "$OUT"/{ref,forge,compare,edge,perf}

H() { printf '\n\033[1;33m═══ %s ═══\033[0m\n' "$*"; }
PY=python3

MODEL=/root/external_test/mobilenetv2-7.onnx
[ -f "$MODEL" ] || { echo "FATAL: $MODEL missing"; exit 1; }
BIN=/root/forge/build-truth/forge
[ -x "$BIN" ] || BIN=/root/forge/build-test/forge
[ -x "$BIN" ] || { echo "FATAL: no forge binary"; exit 1; }

sha256sum "$MODEL" | tee "$OUT/ref/model.sha256"
ls -la "$MODEL" | tee -a "$OUT/ref/model.sha256"

# ─────────────────────────────────────────────────────────────
H "PART A — REFERENCE EXTRACTION (Python onnx)"
# ─────────────────────────────────────────────────────────────

"$PY" - "$MODEL" > "$OUT/ref/ground-truth.json" 2> "$OUT/ref/errors.txt" << 'PYEOF'
import sys, json, hashlib
import onnx
from onnx import numpy_helper
import numpy as np

path = sys.argv[1]
m = onnx.load(path)
g = m.graph

def fnv1a_64(data: bytes) -> str:
    h = 0xcbf29ce484222325
    prime = 0x100000001b3
    mask = 0xFFFFFFFFFFFFFFFF
    for b in data:
        h ^= b
        h = (h * prime) & mask
    return f"{h:016x}"

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
        "input_count": len(n.input),
        "output_count": len(n.output),
    } for n in g.node],
    "initializers": [],
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
}

for init in g.initializer:
    arr = numpy_helper.to_array(init)
    raw = arr.tobytes()
    info["initializers"].append({
        "name": init.name,
        "dtype": int(init.data_type),
        "numpy_dtype": str(arr.dtype),
        "dims": [int(d) for d in init.dims],
        "elem_count": int(arr.size),
        "bytes": len(raw),
        "fnv1a": fnv1a_64(raw),
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
    })

print(json.dumps(info, indent=2, sort_keys=True))
PYEOF

echo "--- summary from Python onnx ---"
"$PY" - << PYEOF
import json
d = json.load(open("$OUT/ref/ground-truth.json"))
print(f"  ir_version  : {d['model']['ir_version']}")
print(f"  producer    : {d['model']['producer_name']} {d['model']['producer_version']}")
print(f"  domain      : '{d['model']['domain']}'")
print(f"  opset       : {d['opset']}")
print(f"  nodes       : {d['graph']['node_count']}")
print(f"  inits       : {d['graph']['initializer_count']}")
print(f"  inputs      : {d['graph']['input_count']}")
print(f"  outputs     : {d['graph']['output_count']}")
print(f"  value_info  : {d['graph']['value_info_count']}")
print()
print("  INPUTS:")
for v in d["inputs"]:
    print(f"    {v['name']:20s} dims={v['dims']} dtype={v['dtype']}")
print("  OUTPUTS:")
for v in d["outputs"]:
    print(f"    {v['name']:20s} dims={v['dims']} dtype={v['dtype']}")
print()
from collections import Counter
ops = Counter(n["op_type"] for n in d["nodes"])
print("  NODE OP TYPES (count):")
for op, c in sorted(ops.items(), key=lambda x: -x[1]):
    print(f"    {op:20s} {c}")
print()
print("  INITIALIZER SUMMARY:")
total_bytes = sum(i["bytes"] for i in d["initializers"])
print(f"    total: {len(d['initializers'])} tensors, {total_bytes} bytes")
print("    first 10:")
for i in d["initializers"][:10]:
    print(f"      {i['name']:40s} dims={i['dims']} dtype={i['numpy_dtype']} bytes={i['bytes']} fnv={i['fnv1a']}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART B — FORGE RUN (--verify-tensors)"
# ─────────────────────────────────────────────────────────────

FORGE_RAW="$OUT/forge/raw.txt"
"$BIN" "$MODEL" --verify-tensors > "$FORGE_RAW" 2>&1
echo "exit: $?"

# ─────────────────────────────────────────────────────────────
H "PART C — TOP-LEVEL FIELD COMPARISON"
# ─────────────────────────────────────────────────────────────

"$PY" - << PYEOF | tee "$OUT/compare/top-level.txt"
import json, re

gt = json.load(open("$OUT/ref/ground-truth.json"))
raw = open("$FORGE_RAW").read()

def grab(p, t):
    m = re.search(p, t)
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
match_count = 0
for f in ["ir_version", "opset", "nodes", "inits", "inputs", "outputs"]:
    m = "OK" if ref[f] == forge[f] else "DIVERGENCE"
    if m == "OK": match_count += 1
    print(f"{f:<14} {ref[f]:<12} {str(forge[f]):<12} {m}")
print()
print(f"matched: {match_count}/6")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART D — INITIALIZER DEEP COMPARISON (name + dims + bytes + FNV-1a)"
# ─────────────────────────────────────────────────────────────

"$PY" - << PYEOF | tee "$OUT/compare/initializers.txt"
import json, re

gt = json.load(open("$OUT/ref/ground-truth.json"))
raw = open("$FORGE_RAW").read()

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

print(f"forge tensors parsed: {len(forge_tensors)}")
print(f"reference initializers: {len(gt['initializers'])}")
print()

# Reference: f32 → "f32", dtype codes
DTYPE_MAP = {1: "f32", 10: "f16", 11: "f64", 6: "i32", 7: "i64", 2: "u8", 3: "i8", 4: "u16", 5: "i16", 9: "bool"}

name_w = 45
print(f"{'NAME':<{name_w}} {'REF_DIMS':<20} {'REF_BYTES':<10} {'FRG_BYTES':<10} {'REF_FNV':<18} {'FRG_FNV':<18} {'RESULT'}")
print("-" * 145)

stats = {"total": 0, "missing": 0, "bytes_match": 0, "hash_match": 0, "hash_differ": 0}
differing = []

for i in gt["initializers"]:
    stats["total"] += 1
    ft = forge_tensors.get(i["name"])
    ref_dims_str = ",".join(str(d) for d in i["dims"])
    if not ft:
        print(f"{i['name']:<{name_w}} {ref_dims_str:<20} {i['bytes']:<10} {'MISSING':<10} {i['fnv1a']:<18} {'-':<18} MISSING")
        stats["missing"] += 1
        continue

    bytes_ok = (i["bytes"] == ft["bytes"])
    hash_ok  = (i["fnv1a"] == ft["hash"])
    if bytes_ok: stats["bytes_match"] += 1
    if hash_ok: stats["hash_match"] += 1
    else:
        stats["hash_differ"] += 1
        differing.append((i["name"], i["fnv1a"], ft["hash"]))

    result = "OK" if (bytes_ok and hash_ok) else ("BYTES_DIFF" if not bytes_ok else "HASH_DIFF")
    print(f"{i['name']:<{name_w}} {ref_dims_str:<20} {i['bytes']:<10} {ft['bytes']:<10} {i['fnv1a']:<18} {ft['hash']:<18} {result}")

print()
print(f"total:        {stats['total']}")
print(f"missing:      {stats['missing']}")
print(f"bytes_match:  {stats['bytes_match']}")
print(f"hash_match:   {stats['hash_match']}")
print(f"hash_differ:  {stats['hash_differ']}")

if differing:
    print()
    print("first 20 hash differences:")
    for name, ref_h, frg_h in differing[:20]:
        print(f"  {name}: ref={ref_h} forge={frg_h}")

# Also: any forge tensor NOT in reference?
ref_names = {i["name"] for i in gt["initializers"]}
extra = [n for n in forge_tensors if n not in ref_names]
print()
print(f"forge-only tensors (not in reference initializers): {len(extra)}")
for e in extra[:20]:
    print(f"  {e}: {forge_tensors[e]}")
PYEOF

# ─────────────────────────────────────────────────────────────
H "PART E — EDGE CASES"
# ─────────────────────────────────────────────────────────────

EDGE="$OUT/edge"

# E.1 truncated to 50%
head -c $(($(stat -c%s "$MODEL") / 2)) "$MODEL" > "$EDGE/truncated-50.onnx"
"$BIN" "$EDGE/truncated-50.onnx" > "$EDGE/truncated-50.out" 2>&1
echo "truncated-50 exit: $?" >> "$EDGE/summary.txt"
echo "truncated-50 stdout_bytes: $(wc -c < "$EDGE/truncated-50.out")" >> "$EDGE/summary.txt"

# E.2 truncated to header only (first 100 bytes)
head -c 100 "$MODEL" > "$EDGE/truncated-100.onnx"
"$BIN" "$EDGE/truncated-100.onnx" > "$EDGE/truncated-100.out" 2>&1
echo "truncated-100 exit: $?" >> "$EDGE/summary.txt"

# E.3 valid header + garbage
{ head -c 500 "$MODEL"; head -c 5000 /dev/urandom; } > "$EDGE/header-garbage.onnx"
"$BIN" "$EDGE/header-garbage.onnx" > "$EDGE/header-garbage.out" 2>&1
echo "header-garbage exit: $?" >> "$EDGE/summary.txt"

# E.4 zero-length
: > "$EDGE/zero.onnx"
"$BIN" "$EDGE/zero.onnx" > "$EDGE/zero.out" 2>&1
echo "zero exit: $?" >> "$EDGE/summary.txt"

# E.5 random bytes 14MB (same size as model)
head -c 14665812 /dev/urandom > "$EDGE/random-14mb.onnx"
"$BIN" "$EDGE/random-14mb.onnx" > "$EDGE/random-14mb.out" 2>&1
echo "random-14mb exit: $?" >> "$EDGE/summary.txt"
echo "random-14mb stdout_bytes: $(wc -c < "$EDGE/random-14mb.out")" >> "$EDGE/summary.txt"

# E.6 non-existent
"$BIN" /nonexistent-xyz-stage-3c.onnx > "$EDGE/nonexistent.out" 2>&1
echo "nonexistent exit: $?" >> "$EDGE/summary.txt"

echo "--- edge cases summary ---"
cat "$EDGE/summary.txt"

echo
echo "--- edge case outputs (first 15 lines each) ---"
for f in "$EDGE"/*.out; do
  echo "### $(basename $f)"
  head -15 "$f" | sed 's/^/  /'
  echo
done

# ─────────────────────────────────────────────────────────────
H "PART F — PERFORMANCE (MobileNetV2, 14 MB)"
# ─────────────────────────────────────────────────────────────

if command -v hyperfine >/dev/null; then
  hyperfine --warmup 5 --runs 30 --shell=none \
    --export-json "$OUT/perf/hyperfine.json" \
    "$BIN $MODEL" \
    > "$OUT/perf/hyperfine.txt" 2>&1
  echo "--- perf ---"
  cat "$OUT/perf/hyperfine.txt"
fi

# /usr/bin/time -v for memory
/usr/bin/time -v "$BIN" "$MODEL" > /dev/null 2> "$OUT/perf/time-memory.txt"
grep -E 'Maximum resident|Elapsed|User time|System time' "$OUT/perf/time-memory.txt" | tee "$OUT/perf/memory-summary.txt"

# ─────────────────────────────────────────────────────────────
H "MANIFEST"
# ─────────────────────────────────────────────────────────────

find "$OUT" -type f -exec sha256sum {} \; > "$OUT/manifest.sha256"
echo "artifact: $OUT"
du -sh "$OUT"
echo "files: $(find "$OUT" -type f | wc -l)"
