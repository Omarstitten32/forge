cmake_minimum_required(VERSION 3.25)

# Fix Boost policy warning
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

# Boost: header-only usage. Verify headers exist but don't link.
find_path(BOOST_INCLUDE_DIR boost/version.hpp)
if(NOT BOOST_INCLUDE_DIR)
    message(WARNING "Boost headers not found — some features may be disabled")
endif()

# Protobuf: not used in Phase 0. Enable in Phase 1.
# find_package(Protobuf REQUIRED)

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
# forge-frontend
# ==========================================================
add_library(forge-frontend STATIC
    src/frontend/onnx_parser.cpp
)
target_include_directories(forge-frontend PUBLIC include)
target_link_libraries(forge-frontend PUBLIC forge-ir)

# ==========================================================
# forge-optimizer
# ==========================================================
add_library(forge-optimizer STATIC
    src/optimizer/optimizer.cpp
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
# CLI executable
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
    )
    target_link_libraries(forge-tests PRIVATE
        forge-core
        forge-ir
        Catch2::Catch2WithMain
    )
    include(CTest)
    include(Catch)
    catch_discover_tests(forge-tests)
endif()

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
message(STATUS "════════════════════════════════════════════════")
message(STATUS "")
