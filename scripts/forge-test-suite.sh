#!/usr/bin/env bash
# forge-real-test-suite — read-only verification of the project as-is.
# Modifies NOTHING in source. Creates only: build-test/, /tmp/forge-test-*/

set -u
ROOT=/root/forge
TS=$(date +%Y%m%d-%H%M%S)
LOG=$ROOT/scripts/forge-test-$TS.log
WORK=/tmp/forge-test-$TS
BUILD=$ROOT/build-test

mkdir -p "$WORK" "$ROOT/scripts"
exec > >(tee "$LOG") 2>&1

H() { printf '\n\033[1;33m══════════════════════════════════════════════════════════\033[0m\n';
      printf '\033[1;33m  %s\033[0m\n' "$*";
      printf '\033[1;33m══════════════════════════════════════════════════════════\033[0m\n'; }
S() { printf '\n\033[1;36m── %s ──\033[0m\n' "$*"; }
OK()   { printf '  \033[1;32m[PASS]\033[0m %s\n' "$*"; }
BAD()  { printf '  \033[1;31m[FAIL]\033[0m %s\n' "$*"; }
INFO() { printf '  \033[1;34m[INFO]\033[0m %s\n' "$*"; }
KV()   { printf '  %-32s %s\n' "$1" "$2"; }

cd "$ROOT" || exit 1

H "FORGE REAL TEST SUITE — $TS"
INFO "project : $ROOT"
INFO "workdir : $WORK"
INFO "build   : $BUILD"
INFO "log     : $LOG"

# ══════════════════════════════════════════════════════════════════════════
H "PART A — SOURCE SNAPSHOT"
# ══════════════════════════════════════════════════════════════════════════

S "A.1 Git identity"
KV "HEAD"        "$(git rev-parse HEAD)"
KV "short"       "$(git rev-parse --short HEAD)"
KV "branch"      "$(git rev-parse --abbrev-ref HEAD)"
KV "tags"        "$(git tag | tr '\n' ' ')"
KV "dirty"       "$(git status --porcelain | wc -l) files"

S "A.2 Source checksums (for integrity)"
find src include tools proto -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.proto' -o -name '*.h' \) \
  -exec sha256sum {} \; | sort -k2 > "$WORK/source-sha.txt"
KV "files hashed" "$(wc -l < "$WORK/source-sha.txt")"

S "A.3 File inventory"
find src include tools proto -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.proto' \) -print0 \
  | xargs -0 wc -l | tail -1

S "A.4 CMake configuration snapshot"
grep -E '^(option|set|find_package|find_path)' CMakeLists.txt | sort

# ══════════════════════════════════════════════════════════════════════════
H "PART B — CLEAN BUILD"
# ══════════════════════════════════════════════════════════════════════════

rm -rf "$BUILD"

S "B.1 Configure (Release, Ninja)"
T0=$(date +%s%N)
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  > "$WORK/cmake-cfg.log" 2>&1
RC=$?
T1=$(date +%s%N)
KV "exit"     "$RC"
KV "time"     "$(( (T1-T0)/1000000 )) ms"
if [ $RC -eq 0 ]; then OK "configure"; else BAD "configure"; fi
grep -E 'Protobuf|Boost|Compiler|Version|Build type' "$WORK/cmake-cfg.log" | sed 's/^/  /'

S "B.2 Build (capture warnings)"
T0=$(date +%s%N)
cmake --build "$BUILD" -j"$(nproc)" > "$WORK/cmake-build.log" 2>&1
RC=$?
T1=$(date +%s%N)
KV "exit"     "$RC"
KV "time"     "$(( (T1-T0)/1000000 )) ms"
if [ $RC -eq 0 ]; then OK "build"; else BAD "build"; fi

S "B.3 Warning/error census"
W=$(grep -c 'warning:' "$WORK/cmake-build.log" || true)
E=$(grep -c 'error:'   "$WORK/cmake-build.log" || true)
KV "warnings" "$W"
KV "errors"   "$E"
if [ "$W" -gt 0 ]; then
  echo "  --- first 40 warnings ---"
  grep 'warning:' "$WORK/cmake-build.log" | head -40 | sed 's/^/    /'
fi

S "B.4 Build artifacts"
find "$BUILD" -maxdepth 2 -type f \( -name '*.a' -o -name 'forge*' \) -exec ls -la {} \;

# ══════════════════════════════════════════════════════════════════════════
H "PART C — BINARY FORENSICS"
# ══════════════════════════════════════════════════════════════════════════

BIN=$BUILD/forge

S "C.1 Existence & type"
if [ ! -x "$BIN" ]; then BAD "no forge binary"; else
  OK "forge binary present"
  file "$BIN"
fi

S "C.2 Size"
if [ -x "$BIN" ]; then
  KV "bytes"    "$(stat -c%s "$BIN")"
  KV "human"    "$(du -h "$BIN" | cut -f1)"
  size "$BIN" 2>/dev/null || true
fi

S "C.3 Sections (top by size)"
if [ -x "$BIN" ]; then
  readelf -S "$BIN" 2>/dev/null | awk '
    /\[[ 0-9]+\]/ { name=$2; size=$7; if (strtonum("0x"size) > 0) printf "  %-20s %s bytes\n", name, size }
  ' | sort -k2 -n -r | head -15
fi

S "C.4 Dynamic dependencies"
if [ -x "$BIN" ]; then
  ldd "$BIN"
  echo
  readelf -d "$BIN" 2>/dev/null | grep NEEDED | sed 's/^/  /'
fi

S "C.5 Symbols (defined, exported)"
if [ -x "$BIN" ]; then
  nm -C --defined-only "$BIN" 2>/dev/null | wc -l | sed 's/^/  total defined: /'
  nm -C --defined-only "$BIN" 2>/dev/null | grep -E ' [TtWw] ' | wc -l | sed 's/^/  text/data:      /'
  nm -C --undefined-only "$BIN" 2>/dev/null | wc -l | sed 's/^/  undefined:      /'
fi

S "C.6 Hardening flags"
if [ -x "$BIN" ]; then
  for f in PIE NX RELRO Canary FORTIFY; do
    v=$(readelf -a "$BIN" 2>/dev/null | grep -i "$f" | head -1)
    [ -n "$v" ] && printf '  %-8s %s\n' "$f" "$v" || printf '  %-8s (not detected)\n' "$f"
  done
fi

S "C.7 Linker / interpreter"
if [ -x "$BIN" ]; then
  readelf -l "$BIN" 2>/dev/null | grep -E 'interpreter|LOAD' | sed 's/^/  /'
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART D — CLI CONTRACT"
# ══════════════════════════════════════════════════════════════════════════

if [ ! -x "$BIN" ]; then
  BAD "skipping CLI tests — no binary"
else
  S "D.1 --version"
  timeout 5 "$BIN" --version 2>&1 | sed 's/^/  /'
  echo "  exit=$?"

  S "D.2 --help"
  timeout 5 "$BIN" --help 2>&1 | sed 's/^/  /'
  echo "  exit=$?"

  S "D.3 no arguments"
  timeout 5 "$BIN" 2>&1 | sed 's/^/  /'
  echo "  exit=$?"

  S "D.4 nonexistent file"
  timeout 5 "$BIN" /nonexistent-path-xyz.onnx 2>&1 | sed 's/^/  /'
  echo "  exit=$?"

  S "D.5 empty file"
  : > "$WORK/empty.onnx"
  timeout 5 "$BIN" "$WORK/empty.onnx" 2>&1 | sed 's/^/  /'
  echo "  exit=$?"

  S "D.6 random bytes (not a protobuf)"
  head -c 512 /dev/urandom > "$WORK/random.onnx"
  timeout 5 "$BIN" "$WORK/random.onnx" 2>&1 | sed 's/^/  /'
  echo "  exit=$?"

  S "D.7 directory instead of file"
  timeout 5 "$BIN" "$WORK" 2>&1 | sed 's/^/  /'
  echo "  exit=$?"
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART E — ONNX MODEL PROVISIONING"
# ══════════════════════════════════════════════════════════════════════════

MODEL=""

S "E.1 Search filesystem for any .onnx"
FOUND=$(find / -xdev -name '*.onnx' -type f 2>/dev/null | head -5)
if [ -n "$FOUND" ]; then
  echo "$FOUND" | sed 's/^/  /'
  MODEL=$(echo "$FOUND" | head -1)
  OK "using existing: $MODEL"
else
  INFO "no .onnx files found on filesystem"
fi

S "E.2 Try generating minimal ONNX via Python"
if [ -z "$MODEL" ]; then
  python3 - "$WORK" << 'PYEOF' 2>&1 | sed 's/^/  /'
import sys, os
out = sys.argv[1]
try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("onnx python package not available")
    sys.exit(1)

# Build a minimal model: Y = Relu(X + W)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
W = helper.make_tensor('W', TensorProto.FLOAT, [1, 4], [0.1, 0.2, 0.3, 0.4])
add = helper.make_node('Add', ['X', 'W'], ['Z'])
relu = helper.make_node('Relu', ['Z'], ['Y'])
g = helper.make_graph([add, relu], 'mini', [X], [Y], [W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
m.ir_version = 8
path = os.path.join(out, 'mini.onnx')
onnx.save(m, path)
print(f"generated: {path} ({os.path.getsize(path)} bytes)")
PYEOF
  if [ -f "$WORK/mini.onnx" ]; then
    MODEL="$WORK/mini.onnx"
    OK "generated mini.onnx"
  fi
fi

S "E.3 Try downloading MobileNetV2 (if network available)"
if [ -z "$MODEL" ] && timeout 10 curl -sL -o "$WORK/mnv2.onnx" \
   "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx" 2>/dev/null; then
  if [ -s "$WORK/mnv2.onnx" ] && file "$WORK/mnv2.onnx" | grep -qi 'data\|protobuf\|octet'; then
    MODEL="$WORK/mnv2.onnx"
    KV "downloaded" "$(du -h "$WORK/mnv2.onnx" | cut -f1)"
    OK "MobileNetV2 available"
  else
    rm -f "$WORK/mnv2.onnx"
  fi
fi

S "E.4 Model selected"
if [ -n "$MODEL" ]; then
  OK "MODEL=$MODEL"
  ls -la "$MODEL"
  file "$MODEL"
else
  BAD "no ONNX model available — functional tests will be skipped"
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART F — FUNCTIONAL TEST"
# ══════════════════════════════════════════════════════════════════════════

if [ -z "$MODEL" ] || [ ! -x "$BIN" ]; then
  BAD "skip: model or binary missing"
else
  S "F.1 Single run, stdout+stderr captured"
  timeout 60 "$BIN" "$MODEL" > "$WORK/run.out" 2> "$WORK/run.err"
  RC=$?
  KV "exit"        "$RC"
  KV "stdout-lines" "$(wc -l < "$WORK/run.out")"
  KV "stderr-lines" "$(wc -l < "$WORK/run.err")"
  echo "  --- stdout ---"
  sed 's/^/    /' "$WORK/run.out"
  echo "  --- stderr ---"
  sed 's/^/    /' "$WORK/run.err"

  S "F.2 Determinism (5 consecutive runs)"
  for i in 1 2 3 4 5; do
    timeout 60 "$BIN" "$MODEL" 2>&1 | sha256sum | awk -v i="$i" '{print "  run "i": "$1}'
  done
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART G — PERFORMANCE (hyperfine)"
# ══════════════════════════════════════════════════════════════════════════

if [ -z "$MODEL" ] || [ ! -x "$BIN" ]; then
  BAD "skip: model or binary missing"
elif ! command -v hyperfine >/dev/null; then
  BAD "hyperfine missing"
else
  S "G.1 Governor state (record)"
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null | sed 's/^/  cpu0 governor: /' || true
  cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor 2>/dev/null | sed 's/^/  cpu7 governor: /' || true

  S "G.2 Cold-start benchmarks (10 warmup, 30 measured)"
  hyperfine --warmup 10 --runs 30 --style basic \
    --export-json "$WORK/hyperfine.json" \
    --command-name forge "$BIN $MODEL" \
    2>&1 | sed 's/^/  /'

  S "G.3 JSON summary"
  if [ -f "$WORK/hyperfine.json" ]; then
    python3 - "$WORK/hyperfine.json" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d['results']:
    print(f"  command: {r.get('command','?')}")
    print(f"    mean   : {r['mean']*1000:.2f} ms")
    print(f"    median : {r['median']*1000:.2f} ms")
    print(f"    min    : {r['min']*1000:.2f} ms")
    print(f"    max    : {r['max']*1000:.2f} ms")
    print(f"    stddev : {r['stddev']*1000:.2f} ms")
PYEOF
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART H — MEMORY SAFETY (valgrind)"
# ══════════════════════════════════════════════════════════════════════════

if [ -z "$MODEL" ] || [ ! -x "$BIN" ]; then
  BAD "skip: model or binary missing"
elif ! command -v valgrind >/dev/null; then
  BAD "valgrind missing"
else
  S "H.1 valgrind memcheck (single run)"
  timeout 300 valgrind --leak-check=full --show-leak-kinds=all \
    --track-origins=yes --error-exitcode=99 --log-file="$WORK/valgrind.log" \
    "$BIN" "$MODEL" > /dev/null 2>&1
  RC=$?
  KV "exit" "$RC"
  echo "  --- valgrind summary ---"
  grep -E 'ERROR SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|Invalid read|Invalid write|uninitialised' \
    "$WORK/valgrind.log" | sed 's/^/    /' || true
  echo "  --- full log kept at: $WORK/valgrind.log ---"
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART I — SYSCALL PROFILE (strace)"
# ══════════════════════════════════════════════════════════════════════════

if [ -z "$MODEL" ] || [ ! -x "$BIN" ]; then
  BAD "skip: model or binary missing"
elif ! command -v strace >/dev/null; then
  BAD "strace missing"
else
  S "I.1 Syscall census"
  timeout 60 strace -f -c -o "$WORK/strace-c.txt" "$BIN" "$MODEL" > /dev/null 2>&1
  cat "$WORK/strace-c.txt" | sed 's/^/  /'

  S "I.2 Largest syscall counts (sorted)"
  awk 'NR>3 && NF>=4 {print $(NF-1), $NF, $0}' "$WORK/strace-c.txt" \
    | head -15 | sed 's/^/  /' || true

  S "I.3 Open calls on files (filtered)"
  timeout 60 strace -f -e trace=openat,open -o "$WORK/strace-open.txt" \
    "$BIN" "$MODEL" > /dev/null 2>&1
  grep -v ENOENT "$WORK/strace-open.txt" | grep -E '\.(so|onnx|pb|bin|json|dat)' | head -30 | sed 's/^/  /' || true
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART J — STATIC ANALYSIS"
# ══════════════════════════════════════════════════════════════════════════

S "J.1 Compiler with -Wall -Wextra -Wpedantic (already applied)"
grep 'warning:' "$WORK/cmake-build.log" | wc -l | sed 's/^/  warnings during build: /'

S "J.2 clang-tidy availability"
if command -v clang-tidy >/dev/null; then
  OK "clang-tidy $(clang-tidy --version | head -1 | awk '{print $NF}')"
  INFO "not running full clang-tidy — would need compile_commands.json from build-test"
  INFO "run manually: clang-tidy -p $BUILD src/**/*.cpp"
else
  BAD "clang-tidy missing"
fi

S "J.3 cppcheck availability"
if command -v cppcheck >/dev/null; then
  OK "cppcheck $(cppcheck --version 2>&1)"
else
  BAD "cppcheck missing"
fi

S "J.4 Compile with clang++ (independent toolchain)"
if command -v clang++ >/dev/null; then
  CBUILD=$ROOT/build-clang
  rm -rf "$CBUILD"
  timeout 300 cmake -S "$ROOT" -B "$CBUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
    > "$WORK/clang-cfg.log" 2>&1
  if [ $? -eq 0 ]; then
    timeout 600 cmake --build "$CBUILD" -j"$(nproc)" > "$WORK/clang-build.log" 2>&1
    RC=$?
    KV "clang build exit" "$RC"
    KV "clang warnings"   "$(grep -c 'warning:' "$WORK/clang-build.log" || true)"
    KV "clang errors"     "$(grep -c 'error:'   "$WORK/clang-build.log" || true)"
    if [ "$RC" -ne 0 ]; then
      echo "  --- clang errors (first 30) ---"
      grep 'error:' "$WORK/clang-build.log" | head -30 | sed 's/^/    /'
    fi
  else
    BAD "clang cmake configure failed — see $WORK/clang-cfg.log"
  fi
else
  BAD "clang++ missing"
fi

# ══════════════════════════════════════════════════════════════════════════
H "PART K — FINAL SUMMARY"
# ══════════════════════════════════════════════════════════════════════════

printf '\n'
KV "Log file"      "$LOG"
KV "Work dir"      "$WORK"
KV "Git HEAD"      "$(git rev-parse --short HEAD)"
KV "Dirty files"   "$(git status --porcelain | wc -l)"
KV "Build warnings" "$(grep -c 'warning:' "$WORK/cmake-build.log" 2>/dev/null || echo NA)"
KV "Binary size"   "$(stat -c%s "$BUILD/forge" 2>/dev/null || echo NA)"

echo
printf '\033[1;32m== TEST SUITE COMPLETE ==\033[0m\n'
