#!/usr/bin/env bash
# Forge core audit - Batch 1: State, Public API, CMake, CI
set -u

ROOT=/root/forge
OUT=$ROOT/scripts/core-audit-1-$(date +%Y%m%d-%H%M).log
exec > >(tee "$OUT") 2>&1

H() { printf '\n\033[1;33m══════════════════════════════════════════════════════════\033[0m\n';
      printf '\033[1;33m  %s\033[0m\n' "$*";
      printf '\033[1;33m══════════════════════════════════════════════════════════\033[0m\n'; }
S() { printf '\n\033[1;36m── %s ──\033[0m\n' "$*"; }
F() { printf '\n\033[1;32m▸▸▸ %s\033[0m\n' "$*"; }

cd "$ROOT" || exit 1

H "1. REPOSITORY STATE"
S "branch"
git rev-parse --abbrev-ref HEAD
S "HEAD"
git rev-parse HEAD
S "last 20 commits"
git log --oneline -20
S "full log with dates and authors"
git log --format='%h | %ad | %an | %s' --date=short -20
S "dirty files (porcelain)"
git status --porcelain
S "branches (all)"
git branch -a
S "tags"
git tag
S "remotes"
git remote -v
S "tracked file count"
git ls-files | wc -l
S "tracked files (list)"
git ls-files
S ".gitignore contents"
cat .gitignore 2>/dev/null || echo MISSING

H "2. PUBLIC API - COMPLETE DUMP"
for f in $(git ls-files 'include/**/*.hpp' 'include/**/*.h' | sort); do
  F "$f"
  printf 'line-count: '
  wc -l "$f" | awk '{print $1}'
  echo "----"
  cat "$f"
  echo "---- END OF $f ----"
  echo
done

H "3. CMAKE STRUCTURE"
S "root CMakeLists.txt"
cat CMakeLists.txt
S "other cmake files (excluding build/ and third_party/)"
find . -name '*.cmake' -not -path './build/*' -not -path './third_party/*' -not -path './.git/*'

H "4. CI/CD WORKFLOWS"
S "workflow files"
find .github -type f 2>/dev/null || echo "no .github directory"
S "full contents of every workflow"
for f in $(find .github -type f 2>/dev/null | sort); do
  F "$f"
  cat "$f"
  echo
done

H "5. DOCS AND SCRIPTS"
S "docs/ contents"
find docs -type f 2>/dev/null -exec echo "--- {} ---" \; -exec cat {} \;
S "scripts/ contents"
ls -la scripts/
for f in $(find scripts -maxdepth 1 -type f -name '*.sh' 2>/dev/null | sort); do
  F "$f"
  head -50 "$f"
done
S "cmake/ directory"
ls -la cmake/ 2>/dev/null || echo "empty or missing"
S "benchmarks/ directory"
ls -la benchmarks/ 2>/dev/null
S "examples/ directory"
ls -la examples/ 2>/dev/null
S "test_data/ directory"
ls -la test_data/ 2>/dev/null
S "tools/ directory tree"
find tools -type f 2>/dev/null

H "6. TESTS PRESENCE"
S "test directories"
find . -path ./build -prune -o -path ./.git -prune -o -type d -name 'test*' -print 2>/dev/null
S "test files"
find . -path ./build -prune -o -path ./.git -prune -o \
  \( -name '*test*.cpp' -o -name '*test*.hpp' -o -name 'test_*.py' \) -print 2>/dev/null

H "7. PROTO FILE SIZE AND SHAPE"
S "line count"
wc -l proto/onnx.proto
S "messages defined"
grep -cE '^message\s+' proto/onnx.proto
S "enums defined"
grep -cE '^enum\s+' proto/onnx.proto
S "first 40 message names"
grep -E '^message\s+' proto/onnx.proto | head -40
S "opset/version references"
grep -inE 'opset|ir_version|version' proto/onnx.proto | head -15

H "8. UNTRACKED / IGNORED STATE"
S "untracked files"
git ls-files --others --exclude-standard
S "ignored files summary"
git status --ignored --porcelain | head -30

H "9. FRESH LOC BREAKDOWN"
find include src tools -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
  | xargs -0 wc -l | sort -n

printf '\n\033[1;32m== BATCH 1 COMPLETE ==\033[0m\n'
printf 'Log saved to: %s\n' "$OUT"
