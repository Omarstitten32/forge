#!/usr/bin/env bash
# forge-stage-1 — extract immutable ground truth. No judgment.
# Output: /root/forge/artifacts/gt-<sha>-<ts>/
set -u

ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
SHA=$(git -C "$ROOT" rev-parse --short HEAD)
DIRTY_HASH=$(git -C "$ROOT" status --porcelain | sha256sum | cut -c1-12)
ART="$ROOT/artifacts/gt-$SHA-$TS-$DIRTY_HASH"
BUILD="$ROOT/build-truth"

mkdir -p "$ART"/{raw,cli,codegen,perf,memory,meta}
exec > >(tee "$ART/run.log") 2>&1

echo "artifact: $ART"
echo "sha: $SHA  dirty-hash: $DIRTY_HASH"
date -Iseconds

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — META (git, env, toolchain)
# ══════════════════════════════════════════════════════════════════

{
  echo "== GIT =="
  git -C "$ROOT" rev-parse HEAD
  git -C "$ROOT" status --porcelain
  git -C "$ROOT" log -1 --format='%H|%ci|%s'
  echo
  echo "== SOURCE HASHES =="
  find "$ROOT"/{src,include,tools,proto} -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.proto' -o -name '*.h' \) \
    -exec sha256sum {} \; | sort -k2
  echo
  echo "== TOOLCHAIN =="
  g++ --version | head -1
  clang++ --version | head -1
  cmake --version | head -1
  ninja --version
  protoc --version
  echo
  echo "== CPU / KERNEL =="
  grep -m1 Features /proc/cpuinfo | cut -c1-500
  uname -a
  echo
  echo "== GOVERNOR (before) =="
  for c in 0 4 7; do
    printf 'cpu%s: ' "$c"
    cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || echo NA
  done
} > "$ART/meta/env.txt"

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD (from scratch, single-threaded for determinism)
# ══════════════════════════════════════════════════════════════════

rm -rf "$BUILD"
START=$(date +%s.%N)
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  > "$ART/meta/cmake-configure.log" 2>&1
CFG_RC=$?
cmake --build "$BUILD" -j1 > "$ART/meta/cmake-build.log" 2>&1
BUILD_RC=$?
END=$(date +%s.%N)

{
  echo "configure_rc: $CFG_RC"
  echo "build_rc:     $BUILD_RC"
  echo "build_time:   $(awk -v s=$START -v e=$END 'BEGIN{printf "%.2f", e-s}') s"
  echo "warnings:     $(grep -c 'warning:' "$ART/meta/cmake-build.log" || echo 0)"
  echo "errors:       $(grep -c 'error:'   "$ART/meta/cmake-build.log" || echo 0)"
  echo
  echo "== WARNINGS (raw) =="
  grep -E 'warning:|error:' "$ART/meta/cmake-build.log" || echo "none"
} > "$ART/meta/build.txt"

BIN="$BUILD/forge"
if [ ! -x "$BIN" ]; then
  echo "FATAL: no binary"; exit 1
fi

sha256sum "$BIN" > "$ART/meta/binary.sha256"
cp "$BIN" "$ART/meta/forge.bin"

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — BINARY FACTS (no judgment, just extraction)
# ══════════════════════════════════════════════════════════════════

{
  echo "== SIZE =="
  stat -c '%s bytes' "$BIN"
  size "$BIN"
  echo
  echo "== TYPE =="
  file "$BIN"
  readelf -h "$BIN" | grep -E 'Class|Data|Machine|Type|Entry'
  echo
  echo "== SECTIONS =="
  readelf -S -W "$BIN" | awk 'NR>5 && NF>=7 {print $2, $3, $6}'
  echo
  echo "== DYNAMIC DEPS =="
  readelf -d "$BIN" | grep NEEDED
  echo
  echo "== INTERP =="
  readelf -l "$BIN" | grep interpreter
  echo
  echo "== STACK FLAGS =="
  readelf -l "$BIN" | grep -A1 GNU_STACK
  echo
  echo "== BIND_NOW / RELRO =="
  readelf -d "$BIN" | grep -E 'BIND_NOW|FLAGS|RELRO' || echo "not present"
  echo
  echo "== STACK PROTECTOR SYMBOL =="
  nm -D "$BIN" 2>/dev/null | grep -c '__stack_chk_fail' || echo 0
  echo
  echo "== FORTIFY SYMBOLS =="
  nm -D "$BIN" 2>/dev/null | grep -c '__.*_chk' || echo 0
  echo
  echo "== TEXT SEGMENT RWX CHECK =="
  readelf -l -W "$BIN" | awk '$1=="LOAD" {print $0}'
} > "$ART/meta/binary.txt"

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — CLI MATRIX (every invocation, raw capture)
# ══════════════════════════════════════════════════════════════════

MODEL=/root/external_test/test_model.onnx
[ -f "$MODEL" ] || { echo "FATAL: no test model"; exit 1; }
sha256sum "$MODEL" > "$ART/meta/model.sha256"

# Prepare synthetic inputs
: > "$ART/raw/empty.onnx"
head -c 512 /dev/urandom > "$ART/raw/random.onnx"
mkdir -p "$ART/raw/dir_input"

run_case() {
  local name="$1"; shift
  local dir="$ART/cli/$name"
  mkdir -p "$dir"

  # 1) exit code + raw stdout/stderr (single run)
  timeout 60 "$BIN" "$@" > "$dir/stdout.raw" 2> "$dir/stderr.raw"
  echo "$?" > "$dir/exit.code"

  # 2) determinism check: 20 runs, normalized stdout hash
  #    normalize: strip timestamp prefix "[HH:MM:SS.mmm] "
  for i in $(seq 1 20); do
    timeout 60 "$BIN" "$@" 2>/dev/null \
      | sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}\] //' \
      | sha256sum | cut -d' ' -f1
  done | sort -u > "$dir/normalized-hashes.txt"

  # 3) peak RSS via /usr/bin/time -v
  /usr/bin/time -v "$BIN" "$@" > /dev/null 2> "$dir/time.txt" || true
  grep -E 'Maximum resident|Elapsed|User time|System time' "$dir/time.txt" \
    > "$dir/time-extracted.txt" || true

  echo "== $name =="
  echo "exit:        $(cat "$dir/exit.code")"
  echo "stdout_size: $(wc -c < "$dir/stdout.raw")"
  echo "stderr_size: $(wc -c < "$dir/stderr.raw")"
  echo "unique_out:  $(wc -l < "$dir/normalized-hashes.txt") (1 = deterministic)"
  cat "$dir/time-extracted.txt" | sed 's/^/  /'
  echo
}

{
  run_case version --version
  run_case help --help
  run_case noargs
  run_case nonexistent /nonexistent-xyz.onnx
  run_case empty "$ART/raw/empty.onnx"
  run_case random "$ART/raw/random.onnx"
  run_case directory "$ART/raw/dir_input"
  run_case valid_default "$MODEL"
  run_case backend_neon "$MODEL" -b neon
  run_case backend_sve "$MODEL" -b sve
  run_case backend_apu "$MODEL" -b apu
  run_case backend_vulkan "$MODEL" -b vulkan
  run_case verify_tensors "$MODEL" --verify-tensors
} > "$ART/cli/summary.txt"

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — CODEGEN OUTPUT (the file that -o would produce)
# ══════════════════════════════════════════════════════════════════

for backend in neon sve apu vulkan; do
  OUT="$ART/codegen/out-$backend.c"
  timeout 60 "$BIN" "$MODEL" -b "$backend" -o "$OUT" \
    > "$ART/codegen/log-$backend.txt" 2>&1
  echo "$?" > "$ART/codegen/exit-$backend.txt"
  if [ -s "$OUT" ]; then
    sha256sum "$OUT" > "$ART/codegen/sha-$backend.txt"
    # Try to compile if it looks like C/C++
    if head -c 100 "$OUT" | grep -qiE 'include|#pragma|void|int'; then
      g++ -c -x c++ -std=c++20 -O0 -o /dev/null "$OUT" 2> "$ART/codegen/compile-$backend.err"
      echo "$?" > "$ART/codegen/compile-exit-$backend.txt"
    fi
  fi
done

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — PERFORMANCE (proper statistical method)
# ══════════════════════════════════════════════════════════════════

# Pin to big cores (4-7) and lock frequency if permitted
for c in 4 5 6 7; do
  echo performance > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true
done

hyperfine \
  --warmup 20 --runs 200 --shell=none \
  --export-json "$ART/perf/hyperfine.json" \
  "$BIN $MODEL" \
  > "$ART/perf/hyperfine.txt" 2>&1 || true

# Governor snapshot after
{
  echo "== GOVERNOR AFTER =="
  for c in 0 4 7; do
    printf 'cpu%s: ' "$c"
    cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || echo NA
  done
} >> "$ART/perf/hyperfine.txt"

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — MEMORY (valgrind, full log)
# ══════════════════════════════════════════════════════════════════

valgrind --leak-check=full --show-leak-kinds=all \
  --track-origins=yes --error-exitcode=99 \
  --log-file="$ART/memory/valgrind-full.log" \
  "$BIN" "$MODEL" > /dev/null 2>&1
echo "$?" > "$ART/memory/valgrind-exit.txt"

# Extract summary lines only (not interpretation)
grep -E 'ERROR SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|Invalid read|Invalid write|uninitialised' \
  "$ART/memory/valgrind-full.log" > "$ART/memory/valgrind-summary.txt" || true

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — RESTORE GOVERNOR
# ══════════════════════════════════════════════════════════════════

for c in 4 5 6 7; do
  echo sugov_ext > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true
done

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — FINAL MANIFEST
# ══════════════════════════════════════════════════════════════════

find "$ART" -type f -exec sha256sum {} \; | sort -k2 > "$ART/manifest.sha256"

cat > "$ART/README.md" << README_EOF
# Ground Truth Snapshot

- **SHA**:        $SHA
- **Dirty hash**: $DIRTY_HASH
- **Timestamp**:  $TS
- **Artifact**:   $ART

## Structure

- \`meta/\`     — git, env, toolchain, build, binary facts
- \`cli/\`      — every CLI invocation (raw stdout/stderr/exit/determinism/time)
- \`codegen/\`  — actual output of \`-o\` for each backend
- \`perf/\`     — hyperfine 200-run measurements
- \`memory/\`   — valgrind full log
- \`manifest.sha256\` — integrity of all files

## Reproduce

    bash /root/forge/scripts/stage-1-ground-truth.sh

## Note

This snapshot contains **raw facts only**. No PASS/FAIL labels.
Interpretation happens in review, not here.
README_EOF

echo
echo "════════════════════════════════════════════════"
echo "ARTIFACT READY: $ART"
echo "════════════════════════════════════════════════"
du -sh "$ART"
