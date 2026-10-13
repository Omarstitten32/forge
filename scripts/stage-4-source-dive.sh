#!/usr/bin/env bash
# stage-4 — full source extraction for parser & codegen analysis. No judgment.
set -u

ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/artifacts/src-dive-$TS"
mkdir -p "$OUT"/{raw,src,stats}

cd "$ROOT" || exit 1

H() { printf '\n\033[1;33m═══ %s ═══\033[0m\n' "$*"; }

# ─────────────────────────────────────────────────────────────
H "PART 1 — RAW OUTPUT FROM STAGE 3C"
# ─────────────────────────────────────────────────────────────

# Find most recent 3c artifact
LAST3C=$(ls -td /root/forge/artifacts/ref-3c-* 2>/dev/null | head -1)
if [ -n "$LAST3C" ] && [ -f "$LAST3C/forge/raw.txt" ]; then
  cp "$LAST3C/forge/raw.txt" "$OUT/raw/mobilenetv2-forge.txt"
  echo "copied from: $LAST3C"
  echo
  echo "--- raw.txt (full) ---"
  cat "$OUT/raw/mobilenetv2-forge.txt"
else
  echo "FATAL: Stage 3C raw.txt not found"
fi

# ─────────────────────────────────────────────────────────────
H "PART 2 — SOURCE EXTRACTION (key files)"
# ─────────────────────────────────────────────────────────────

FILES=(
  "src/frontend/onnx_parser.cpp"
  "src/codegen/codegen.cpp"
  "src/codegen/kernels.cpp"
  "src/optimizer/evaluate.cpp"
  "src/optimizer/fusion.cpp"
  "tools/forge-cli/main.cpp"
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    safe=$(echo "$f" | tr '/' '_')
    cp "$f" "$OUT/src/$safe"
    echo "  $f -> $OUT/src/$safe ($(wc -l < "$f") lines)"
  fi
done

# ─────────────────────────────────────────────────────────────
H "PART 3 — LINE-NUMBERED EXTRACTS (for reference)"
# ─────────────────────────────────────────────────────────────

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    safe=$(echo "$f" | tr '/' '_')
    nl -ba "$f" > "$OUT/src/${safe}.numbered"
  fi
done

# ─────────────────────────────────────────────────────────────
H "PART 4 — FUNCTION MAP"
# ─────────────────────────────────────────────────────────────

{
  for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    echo "=== $f ==="
    grep -nE '^[a-zA-Z_][a-zA-Z0-9_:<>,&*\s]*\s+[a-zA-Z_][a-zA-Z0-9_:]*\s*\(' "$f" \
      | grep -v '^\s*//' \
      | head -60
    echo
  done
} > "$OUT/stats/functions.txt"

cat "$OUT/stats/functions.txt"

# ─────────────────────────────────────────────────────────────
H "PART 5 — OP HANDLING IN PARSER"
# ─────────────────────────────────────────────────────────────

{
  echo "=== op_type string literals in onnx_parser.cpp ==="
  grep -nE '"[A-Z][a-zA-Z]+"' src/frontend/onnx_parser.cpp \
    | grep -vE '"(f32|f16|bf16|i8|i16|i32|i64|u8)"' \
    | head -80
  echo
  echo "=== comparison with OpType enum ==="
  grep -nE '^\s+[A-Z][a-zA-Z]+,' include/forge/ir/op.hpp
} > "$OUT/stats/op-mapping.txt"

cat "$OUT/stats/op-mapping.txt"

# ─────────────────────────────────────────────────────────────
H "PART 6 — SHAPE INFERENCE EVIDENCE"
# ─────────────────────────────────────────────────────────────

{
  echo "=== all references to TensorInfo / set shape in parser ==="
  grep -nE 'TensorInfo|\.shape|shape\(\)|set_shape|infer' src/frontend/onnx_parser.cpp
  echo
  echo "=== all references to dim_value / dim_param / ShapeProto ==="
  grep -nE 'dim_value|dim_param|TensorShapeProto|TypeProto' src/frontend/onnx_parser.cpp
  echo
  echo "=== value_info handling ==="
  grep -nE 'value_info' src/frontend/onnx_parser.cpp
} > "$OUT/stats/shape-inference.txt"

cat "$OUT/stats/shape-inference.txt"

# ─────────────────────────────────────────────────────────────
H "PART 7 — FNV HASH IMPLEMENTATION"
# ─────────────────────────────────────────────────────────────

{
  echo "=== FNV implementation in main.cpp ==="
  grep -nE 'fnv|0xcbf|0x100000001b3|hash' tools/forge-cli/main.cpp
  echo
  echo "=== context lines around fnv ==="
  grep -B2 -A15 -iE 'fnv|fnv1a' tools/forge-cli/main.cpp || echo "no fnv reference"
} > "$OUT/stats/fnv.txt"

cat "$OUT/stats/fnv.txt"

# ─────────────────────────────────────────────────────────────
H "PART 8 — CODEGEN CONTENT"
# ─────────────────────────────────────────────────────────────

{
  echo "=== codegen.cpp full content ==="
  cat src/codegen/codegen.cpp
  echo
  echo "=== kernels.cpp full content ==="
  cat src/codegen/kernels.cpp
} > "$OUT/raw/codegen-content.txt"

echo "saved to: $OUT/raw/codegen-content.txt"
echo "lines: $(wc -l < "$OUT/raw/codegen-content.txt")"

# ─────────────────────────────────────────────────────────────
H "PART 9 — FUSION PATTERNS"
# ─────────────────────────────────────────────────────────────

{
  echo "=== pattern definitions in fusion.cpp ==="
  grep -nE 'Conv|Relu|BatchNorm|Add|Mul|fuse|is_fused|OpType::' src/optimizer/fusion.cpp \
    | head -80
} > "$OUT/stats/fusion-patterns.txt"

cat "$OUT/stats/fusion-patterns.txt"

# ─────────────────────────────────────────────────────────────
H "PART 10 — SUMMARY STATS"
# ─────────────────────────────────────────────────────────────

{
  echo "=== per-file: LOC, functions, classes ==="
  for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    loc=$(wc -l < "$f")
    funcs=$(grep -cE '^[a-zA-Z_].*\s+[a-zA-Z_]+\s*\([^;]*$' "$f")
    printf "  %-40s  %5d LOC  %3d funcs\n" "$f" "$loc" "$funcs"
  done
  echo
  echo "=== all 'TODO' / 'FIXME' / 'STUB' / 'Not implemented' in source ==="
  grep -rnE 'TODO|FIXME|STUB|Not implemented|not implemented|placeholder' \
    src include tools --include='*.cpp' --include='*.hpp' | head -40
} > "$OUT/stats/summary.txt"

cat "$OUT/stats/summary.txt"

# ─────────────────────────────────────────────────────────────
H "MANIFEST"
# ─────────────────────────────────────────────────────────────

find "$OUT" -type f -exec sha256sum {} \; > "$OUT/manifest.sha256"
echo "artifact: $OUT"
du -sh "$OUT"
