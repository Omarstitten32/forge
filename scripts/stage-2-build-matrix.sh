#!/usr/bin/env bash
# stage-2-build-matrix — file/compile/build analysis. Raw data only.
set -u

ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/artifacts/build-matrix-$TS"
mkdir -p "$OUT"/{compile,links,analysis,matrix,build}

H() { printf '\n\033[1;33m═══ %s ═══\033[0m\n' "$*"; }

cd "$ROOT" || exit 1

# ─────────────────────────────────────────────────────────────
H "PART 1 — FILE INVENTORY"
# ─────────────────────────────────────────────────────────────
{
  echo "== SOURCE FILES + LOC =="
  find src include tools -type f \( -name '*.cpp' -o -name '*.hpp' \) -print0 \
    | xargs -0 wc -l | sort -n
  echo
  echo "== PER MODULE =="
  for m in core ir frontend optimizer codegen runtime; do
    n=$(find src/$m include/forge/$m -type f \( -name '*.cpp' -o -name '*.hpp' \) 2>/dev/null | wc -l)
    l=$(find src/$m include/forge/$m -type f \( -name '*.cpp' -o -name '*.hpp' \) 2>/dev/null -exec cat {} + | wc -l)
    printf '  %-12s %2d files  %5d lines\n' "$m" "$n" "$l"
  done
  printf '  cli          %d files  %d lines\n' \
    "$(find tools -type f | wc -l)" "$(cat tools/forge-cli/main.cpp | wc -l)"
} > "$OUT/matrix/inventory.txt"
cat "$OUT/matrix/inventory.txt"

# ─────────────────────────────────────────────────────────────
H "PART 2 — INCLUDE GRAPH"
# ─────────────────────────────────────────────────────────────
PROTO_GEN="$ROOT/build-truth/proto-gen"
[ -d "$PROTO_GEN" ] || PROTO_GEN="$ROOT/build-test/proto-gen"
[ -d "$PROTO_GEN" ] || PROTO_GEN="$ROOT/build/proto-gen"
[ -d "$PROTO_GEN" ] || { echo "FATAL: proto-gen not found"; exit 1; }

INCLUDES="-I$ROOT/include -I$PROTO_GEN -I/usr/include/eigen3"

{
  for f in $(find src tools -name '*.cpp' | sort); do
    echo "--- $f ---"
    grep -E '^#include' "$f" | sed 's/^#include //; s/[<>"]//g' | tr '\n' ' '
    echo
  done
} > "$OUT/matrix/includes.txt"
head -40 "$OUT/matrix/includes.txt"

# ─────────────────────────────────────────────────────────────
H "PART 3 — DIRECT COMPILE MATRIX"
# ─────────────────────────────────────────────────────────────

FILES=(
  "src/core/log.cpp"
  "src/ir/graph.cpp"
  "src/optimizer/fusion.cpp"
  "src/codegen/kernels.cpp"
  "src/frontend/onnx_parser.cpp"
  "tools/forge-cli/main.cpp"
)

FLAGS=("-O0" "-O2" "-O3" "-Os")
COMPILERS=("g++" "clang++")

{
  printf "%-8s %-5s %-22s %-12s %s\n" "CC" "OPT" "FILE" "STATUS" "WARN"
  printf "%-8s %-5s %-22s %-12s %s\n" "------" "-----" "----" "------" "----"
  for CC in "${COMPILERS[@]}"; do
    for OPT in "${FLAGS[@]}"; do
      for F in "${FILES[@]}"; do
        LOG="$OUT/compile/${CC}-${OPT}-$(basename $F).log"
        timeout 30 $CC -fsyntax-only -std=c++20 $OPT \
          -Wall -Wextra -Wpedantic -Wno-unused-parameter \
          $INCLUDES "$F" > "$LOG" 2>&1
        RC=$?
        W=$(grep -c 'warning:' "$LOG" 2>/dev/null || echo 0)
        E=$(grep -c 'error:'   "$LOG" 2>/dev/null || echo 0)
        if [ $RC -eq 0 ]; then
          STATUS="OK"
        else
          STATUS="ERR($E)"
        fi
        printf "%-8s %-5s %-22s %-12s %s\n" "$CC" "$OPT" "$(basename $F)" "$STATUS" "$W"
      done
    done
  done
} > "$OUT/matrix/compile-matrix.txt"
cat "$OUT/matrix/compile-matrix.txt"

# ─────────────────────────────────────────────────────────────
H "PART 4 — FULL BUILD × BUILD TYPES"
# ─────────────────────────────────────────────────────────────

for BT in Debug Release; do
  BD="$OUT/build/$BT"
  cmake -S "$ROOT" -B "$BD" -G Ninja -DCMAKE_BUILD_TYPE="$BT" \
    > "$OUT/build/$BT-cfg.log" 2>&1
  T0=$(date +%s.%N)
  cmake --build "$BD" -j"$(nproc)" > "$OUT/build/$BT-build.log" 2>&1
  RC=$?
  T1=$(date +%s.%N)
  DUR=$(awk -v s=$T0 -v e=$T1 'BEGIN{printf "%.2f", e-s}')
  SZ=$(stat -c%s "$BD/forge" 2>/dev/null || echo 0)
  W=$(grep -c 'warning:' "$OUT/build/$BT-build.log" 2>/dev/null || echo 0)
  E=$(grep -c 'error:'   "$OUT/build/$BT-build.log" 2>/dev/null || echo 0)
  printf "  %-10s rc=%d  time=%ss  size=%s  W=%s  E=%s\n" \
    "$BT" "$RC" "$DUR" "$SZ" "$W" "$E"
done | tee "$OUT/matrix/build-matrix.txt"

# ─────────────────────────────────────────────────────────────
H "PART 5 — STATIC ANALYSIS"
# ─────────────────────────────────────────────────────────────

if command -v cppcheck >/dev/null; then
  timeout 120 cppcheck --enable=warning,performance,portability \
    --std=c++20 --suppress=missingIncludeSystem \
    -I include "$ROOT/src" "$ROOT/tools" \
    > "$OUT/analysis/cppcheck.txt" 2>&1
  echo "cppcheck lines: $(wc -l < "$OUT/analysis/cppcheck.txt")"
  echo "--- cppcheck first 30 ---"
  head -30 "$OUT/analysis/cppcheck.txt"
fi

# ─────────────────────────────────────────────────────────────
H "PART 6 — DEAD CODE / SYMBOLS"
# ─────────────────────────────────────────────────────────────

BD="$OUT/build/Release"
{
  echo "== SYMBOLS PER .a =="
  for a in "$BD"/libforge-*.a; do
    [ -f "$a" ] || continue
    n=$(nm -g --defined-only "$a" 2>/dev/null | wc -l)
    sz=$(stat -c%s "$a")
    printf "  %-30s %5d symbols  %8d bytes\n" "$(basename $a)" "$n" "$sz"
  done
  echo
  echo "== FINAL BINARY EXPORTED SYMBOLS =="
  nm -D "$BD/forge" 2>/dev/null | wc -l
} > "$OUT/links/symbols.txt"
cat "$OUT/links/symbols.txt"

# ─────────────────────────────────────────────────────────────
H "PART 7 — DETERMINISTIC BUILD"
# ─────────────────────────────────────────────────────────────

D1="$OUT/build/det1"
D2="$OUT/build/det2"
for D in "$D1" "$D2"; do
  cmake -S "$ROOT" -B "$D" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build "$D" -j"$(nproc)" >/dev/null 2>&1
done

{
  echo "== BINARY SHA256 =="
  sha256sum "$D1/forge" "$D2/forge"
  echo
  if cmp -s "$D1/forge" "$D2/forge"; then
    echo "deterministic: YES"
  else
    echo "deterministic: NO"
    echo "diff bytes: $(cmp -l "$D1/forge" "$D2/forge" 2>/dev/null | wc -l)"
    echo "--- first 10 diffs (offset old new) ---"
    cmp -l "$D1/forge" "$D2/forge" 2>/dev/null | head -10
  fi
} > "$OUT/matrix/deterministic.txt"
cat "$OUT/matrix/deterministic.txt"

# ─────────────────────────────────────────────────────────────
H "PART 8 — CLANG WITH WORKAROUND"
# ─────────────────────────────────────────────────────────────

CD="$OUT/build/clang-workaround"
cmake -S "$ROOT" -B "$CD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-DSPDLOG_USE_STD_FORMAT=1" \
  > "$OUT/build/clang-cfg.log" 2>&1
cmake --build "$CD" -j"$(nproc)" > "$OUT/build/clang-build.log" 2>&1
RC=$?
W=$(grep -c 'warning:' "$OUT/build/clang-build.log" 2>/dev/null || echo 0)
E=$(grep -c 'error:'   "$OUT/build/clang-build.log" 2>/dev/null || echo 0)
{
  echo "exit:     $RC"
  echo "warnings: $W"
  echo "errors:   $E"
  echo
  echo "== FIRST 15 ERRORS =="
  grep 'error:' "$OUT/build/clang-build.log" 2>/dev/null | head -15
} > "$OUT/matrix/clang-workaround.txt"
cat "$OUT/matrix/clang-workaround.txt"

# ─────────────────────────────────────────────────────────────
H "SUMMARY"
# ─────────────────────────────────────────────────────────────

echo "artifact: $OUT"
find "$OUT" -type f -exec sha256sum {} \; > "$OUT/manifest.sha256"
echo "files hashed: $(wc -l < "$OUT/manifest.sha256")"
du -sh "$OUT"
