#!/usr/bin/env bash
# forge-env-audit — read-only environment audit
set -u

SECTION() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ITEM()    { printf '  %-28s %s\n' "$1" "$2"; }

have() { command -v "$1" >/dev/null 2>&1; }
grab() { "$@" 2>/dev/null || echo MISSING; }

SECTION "1. OS / KERNEL / ARCH"
ITEM "uname -s"        "$(uname -s)"
ITEM "uname -r"        "$(uname -r)"
ITEM "uname -m"        "$(uname -m)"
ITEM "arch"            "$(grab uname -p)"
ITEM "libc"            "$(grab ldd --version | head -n1)"
ITEM "distro"          "$(grab grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '"')"
ITEM "kernel-cmdline"  "$(grab cat /proc/cmdline)"
ITEM "hostname"        "$(uname -n)"
ITEM "uptime"          "$(grab uptime -p)"
ITEM "container"       "$( [ -f /.dockerenv ] && echo docker || echo none )"

SECTION "2. CPU"
ITEM "model"           "$(grab grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
ITEM "cores(logical)"  "$(grab nproc)"
ITEM "cores(physical)" "$(grab lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,\"\",\$2);print \$2}')"
ITEM "sockets"         "$(grab lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,\"\",\$2);print \$2}')"
ITEM "threads/core"    "$(grab lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,\"\",\$2);print \$2}')"
ITEM "flags"           "$(grab grep -m1 'Features' /proc/cpuinfo | cut -d: -f2- | xargs | cut -c1-120)"
ITEM "governor"        "$(grab cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
ITEM "cur-freq-kHz"    "$(grab cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
ITEM "max-freq-kHz"    "$(grab cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
ITEM "throttle-bits"   "$(grab vcgencmd get_throttled)"

SECTION "3. MEMORY / STORAGE"
ITEM "mem-total"       "$(grab awk '/MemTotal/{print \$2\" kB\"}' /proc/meminfo)"
ITEM "mem-free"        "$(grab awk '/MemAvailable/{print \$2\" kB\"}' /proc/meminfo)"
ITEM "swap"            "$(grab awk '/SwapTotal/{print \$2\" kB\"}' /proc/meminfo)"
ITEM "root-fs"         "$(grab df -hT / | tail -n1)"
ITEM "root-avail"      "$(grab df -h / | awk 'NR==2{print \$4}')"
ITEM "build-dir-size"  "$(grab du -sh /root/forge/build)"

SECTION "4. COMPILER TOOLCHAIN"
ITEM "gcc"             "$(grab gcc --version | head -n1)"
ITEM "g++"             "$(grab g++ --version | head -n1)"
ITEM "clang"           "$(grab clang --version | head -n1)"
ITEM "clang++"         "$(grab clang++ --version | head -n1)"
ITEM "cc"              "$(grab cc --version | head -n1)"
ITEM "ld"              "$(grab ld --version | head -n1)"
ITEM "lld"             "$(grab ld.lld --version | head -n1)"
ITEM "ar"              "$(grab ar --version | head -n1)"
ITEM "ranlib"          "$(grab ranlib --version | head -n1)"
ITEM "strip"           "$(grab strip --version | head -n1)"
ITEM "objdump"         "$(grab objdump --version | head -n1)"
ITEM "readelf"         "$(grab readelf --version | head -n1)"
ITEM "gcc-target"      "$(grab gcc -dumpmachine)"
ITEM "gcc-search-dirs" "$(grab gcc -print-search-dirs | head -n1)"

SECTION "5. BUILD SYSTEM"
ITEM "cmake"           "$(grab cmake --version | head -n1)"
ITEM "ninja"           "$(grab ninja --version)"
ITEM "make"            "$(grab make --version | head -n1)"
ITEM "pkg-config"      "$(grab pkg-config --version)"
ITEM "ccache"          "$(grab ccache --version | head -n1)"
ITEM "sccache"         "$(grab sccache --version | head -n1)"
ITEM "autoconf"        "$(grab autoconf --version | head -n1)"
ITEM "automake"        "$(grab automake --version | head -n1)"
ITEM "libtool"         "$(grab libtool --version | head -n1)"

SECTION "6. LINKER / BINARY UTILS"
ITEM "ldd"             "$(grab ldd --version | head -n1)"
ITEM "patchelf"        "$(grab patchelf --version)"
ITEM "nm"              "$(grab nm --version | head -n1)"
ITEM "size"            "$(grab size --version | head -n1)"
ITEM "strings"         "$(grab strings --version | head -n1)"
ITEM "eu-readelf"      "$(grab eu-readelf --version | head -n1)"

SECTION "7. DEBUG / PROFILING"
ITEM "gdb"             "$(grab gdb --version | head -n1)"
ITEM "lldb"            "$(grab lldb --version | head -n1)"
ITEM "valgrind"        "$(grab valgrind --version)"
ITEM "perf"            "$(grab perf --version)"
ITEM "strace"          "$(grab strace --version | head -n1)"
ITEM "ltrace"          "$(grab ltrace --version | head -n1)"
ITEM "hyperfine"       "$(grab hyperfine --version)"
ITEM "time(bash)"      "$(type -a time 2>/dev/null | head -n1)"
ITEM "flamegraph"      "$(have flamegraph.pl && echo yes || echo MISSING)"

SECTION "8. LANGUAGES / RUNTIMES"
ITEM "python3"         "$(grab python3 --version)"
ITEM "python3-path"    "$(grab which python3)"
ITEM "pip3"            "$(grab pip3 --version | cut -c1-60)"
ITEM "node"            "$(grab node --version)"
ITEM "npm"             "$(grab npm --version)"
ITEM "rustc"           "$(grab rustc --version)"
ITEM "cargo"           "$(grab cargo --version)"
ITEM "go"              "$(grab go version)"
ITEM "java"            "$(grab java -version 2>&1 | head -n1)"
ITEM "perl"            "$(grab perl --version | sed -n 2p)"
ITEM "ruby"            "$(grab ruby --version)"

SECTION "9. FORGE BUILD DEPENDENCIES (pkg-config)"
for p in protobuf protobuf-lite fmt spdlog eigen3 cli11 catch2 gtest \
         zlib zstd lz4 openssl libcurl; do
  if pkg-config --exists "$p" 2>/dev/null; then
    ITEM "$p" "$(pkg-config --modversion "$p")"
  else
    ITEM "$p" "MISSING"
  fi
done

SECTION "10. HEADER PRESENCE"
for h in \
  /usr/include/google/protobuf/message.h \
  /usr/include/fmt/format.h \
  /usr/include/spdlog/spdlog.h \
  /usr/include/eigen3/Eigen/Core \
  /usr/include/CLI/CLI.hpp \
  /usr/include/boost/version.hpp \
  /usr/include/catch2/catch_test_macros.hpp \
  /usr/include/gtest/gtest.h ; do
  [ -f "$h" ] && ITEM "$(basename $(dirname $(dirname $h)))" "OK  $h" \
              || ITEM "$(basename $(dirname $(dirname $h)))" "MISSING  $h"
done

SECTION "11. PROTOBUF DETAIL"
ITEM "protoc"          "$(grab protoc --version)"
ITEM "protoc-path"     "$(grab which protoc)"
ITEM "libprotobuf.so"  "$(grab ls -l /usr/lib/aarch64-linux-gnu/libprotobuf.so*)"
ITEM "protobuf-pc"     "$(grab pkg-config --modversion protobuf)"
ITEM "protobuf-lite-pc" "$(grab pkg-config --modversion protobuf-lite)"

SECTION "12. SHARED LIBS LINKED BY forge"
if [ -x /root/forge/build/forge ]; then
  ITEM "binary"        "/root/forge/build/forge"
  ITEM "size"          "$(grab du -h /root/forge/build/forge | cut -f1)"
  ITEM "size-bytes"    "$(grab stat -c%s /root/forge/build/forge)"
  ITEM "type"          "$(grab file /root/forge/build/forge)"
  printf '\n  -- ldd --\n'
  ldd /root/forge/build/forge 2>/dev/null | sed 's/^/    /'
else
  ITEM "forge"         "NOT BUILT"
fi

SECTION "13. GIT STATE"
cd /root/forge 2>/dev/null || exit 0
ITEM "branch"          "$(grab git rev-parse --abbrev-ref HEAD)"
ITEM "HEAD"            "$(grab git rev-parse --short HEAD)"
ITEM "last-commit"     "$(grab git log -1 --format='%h %ad %s' --date=short)"
ITEM "commits"         "$(grab git rev-list --count HEAD)"
ITEM "authors"         "$(grab git shortlog -sne HEAD | wc -l)"
ITEM "dirty-files"     "$(grab git status --porcelain | wc -l)"
ITEM "remote"          "$(grab git remote -v | head -n1)"
ITEM "tags"            "$(grab git tag | tr '\n' ' ')"
ITEM "git-hooks"       "$(grab ls /root/forge/.git/hooks | grep -v sample | tr '\n' ' ')"

SECTION "14. CMAKE CACHE SNAPSHOT"
C=/root/forge/build/CMakeCache.txt
if [ -f "$C" ]; then
  for k in CMAKE_BUILD_TYPE CMAKE_CXX_COMPILER CMAKE_CXX_FLAGS \
           CMAKE_CXX_FLAGS_RELEASE CMAKE_EXE_LINKER_FLAGS \
           CMAKE_INTERPROCEDURAL_OPTIMIZATION Protobuf_VERSION \
           fmt_DIR spdlog_DIR Eigen3_DIR CLI11_DIR CMAKE_GENERATOR; do
    v="$(grep -m1 "^${k}:" "$C" 2>/dev/null | cut -d= -f2-)"
    ITEM "$k" "${v:-<unset>}"
  done
else
  ITEM "CMakeCache" "MISSING"
fi

SECTION "15. KERNEL TUNING (perf-relevant)"
ITEM "perf_event_paranoid" "$(grab cat /proc/sys/kernel/perf_event_paranoid)"
ITEM "sched_autogroup"     "$(grab cat /proc/sys/kernel/sched_autogroup_enabled)"
ITEM "thp-enabled"         "$(grab cat /sys/kernel/mm/transparent_hugepage/enabled)"
ITEM "swappiness"          "$(grab cat /proc/sys/vm/swappiness)"
ITEM "numa-nodes"          "$(grab ls /sys/devices/system/node/ | grep -c node)"

SECTION "16. LIMITS"
ITEM "ulimit-nofile"       "$(ulimit -n)"
ITEM "ulimit-stack"        "$(ulimit -s)"
ITEM "ulimit-core"         "$(ulimit -c)"

printf '\n\033[1;32m== AUDIT COMPLETE ==\033[0m\n'
