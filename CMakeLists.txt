cmake_minimum_required(VERSION 3.25)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(Forge
    VERSION 0.1.0
    DESCRIPTION "AI model compiler for ARM64"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================================================
# Static stdlib: OFF by default (increases size when
# dependents like protobuf/spdlog are still dynamic)
# ==========================================================
option(FORGE_STATIC_STDLIB "Statically link libstdc++/libgcc" OFF)
if(FORGE_STATIC_STDLIB AND UNIX AND NOT APPLE)
    add_link_options(-static-libgcc -static-libstdc++)
endif()

# ==========================================================
# Strip symbols in Release builds (safe, always beneficial)
# ==========================================================
option(FORGE_STRIP_BINARY "Strip binary in Release" ON)
if(FORGE_STRIP_BINARY AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_link_options(-s)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")

option(FORGE_BUILD_TESTS "Build tests" ON)
option(FORGE_ENABLE_APU  "Enable APU"  ON)

# ==========================================================
# Dependencies
# ==========================================================
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CLI11 REQUIRED)

if(FORGE_BUILD_TESTS)
    find_package(Catch2 3 REQUIRED)
endif()

# Boost: header-only
find_path(BOOST_INCLUDE_DIR boost/version.hpp)

# Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "  Protobuf      : ${Protobuf_VERSION}")

# ==========================================================
# Generate C++ from onnx.proto3
# ==========================================================
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/onnx.proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto-gen")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

set(ONNX_PB_CC "${PROTO_GEN_DIR}/onnx.pb.cc")
set(ONNX_PB_H  "${PROTO_GEN_DIR}/onnx.pb.h")

add_custom_command(
    OUTPUT "${ONNX_PB_CC}" "${ONNX_PB_H}"
    COMMAND protobuf::protoc
    ARGS
        --cpp_out=${PROTO_GEN_DIR}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
        ${PROTO_FILE}
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating C++ from onnx.proto3"
    VERBATIM
)

# ==========================================================
# forge-core
# ==========================================================
add_library(forge-core STATIC
    src/core/log.cpp
    src/core/error.cpp
)
target_include_directories(forge-core PUBLIC include)
target_link_libraries(forge-core PUBLIC fmt::fmt spdlog::spdlog)

# ==========================================================
# forge-ir
# ==========================================================
add_library(forge-ir STATIC
    src/ir/graph.cpp
    src/ir/tensor.cpp
    src/ir/op.cpp
)
target_include_directories(forge-ir PUBLIC include)
target_link_libraries(forge-ir PUBLIC forge-core Eigen3::Eigen)

# ==========================================================
# forge-frontend (with Protobuf ONNX)
# ==========================================================
add_library(forge-frontend STATIC
    src/frontend/onnx_parser.cpp
    "${ONNX_PB_CC}"
)
target_include_directories(forge-frontend
    PUBLIC include
    PRIVATE "${PROTO_GEN_DIR}"
)
target_link_libraries(forge-frontend PUBLIC
    forge-ir
    protobuf::libprotobuf
)

# ==========================================================
# forge-optimizer
# ==========================================================
add_library(forge-optimizer STATIC
    src/optimizer/optimizer.cpp
    src/optimizer/analysis.cpp
    src/optimizer/evaluate.cpp
    src/optimizer/fusion.cpp
)
target_include_directories(forge-optimizer PUBLIC include)
target_link_libraries(forge-optimizer PUBLIC forge-ir)

# ==========================================================
# forge-codegen
# ==========================================================
add_library(forge-codegen STATIC
    src/codegen/codegen.cpp
)
target_include_directories(forge-codegen PUBLIC include)
target_link_libraries(forge-codegen PUBLIC forge-ir)

# ==========================================================
# forge-runtime
# ==========================================================
add_library(forge-runtime STATIC
    src/runtime/runtime.cpp
)
target_include_directories(forge-runtime PUBLIC include)
target_link_libraries(forge-runtime PUBLIC forge-ir forge-codegen Threads::Threads)

# ==========================================================
# CLI
# ==========================================================
add_executable(forge tools/forge-cli/main.cpp)
target_link_libraries(forge PRIVATE
    forge-core
    forge-ir
    forge-frontend
    forge-optimizer
    forge-codegen
    forge-runtime
    CLI11::CLI11
)

# ==========================================================
# Tests
# ==========================================================
if(FORGE_BUILD_TESTS)
    enable_testing()
    add_executable(forge-tests
        tests/unit/test_core.cpp
        tests/unit/test_ir.cpp
        tests/unit/test_optimizer.cpp
        tests/unit/test_fusion.cpp
    )
    target_link_libraries(forge-tests PRIVATE
        forge-core
        forge-ir
        forge-optimizer
        forge-frontend
        Catch2::Catch2WithMain
    )
    include(CTest)
    include(Catch)
    catch_discover_tests(forge-tests)
endif()

# ==========================================================
# Summary
# ==========================================================
message(STATUS "")
message(STATUS "════════════════════════════════════════════════")
message(STATUS "  FORGE Configuration Summary")
message(STATUS "════════════════════════════════════════════════")
message(STATUS "  Version       : ${PROJECT_VERSION}")
message(STATUS "  Compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests         : ${FORGE_BUILD_TESTS}")
message(STATUS "  APU backend   : ${FORGE_ENABLE_APU}")
message(STATUS "  Boost headers : ${BOOST_INCLUDE_DIR}")
message(STATUS "  Protobuf      : ${Protobuf_VERSION}")
message(STATUS "  Proto file    : ${PROTO_FILE}")
message(STATUS "════════════════════════════════════════════════")
message(STATUS "")
