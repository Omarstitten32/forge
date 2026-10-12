#pragma once

#include "forge/ir/graph.hpp"
#include <string>

namespace forge::codegen {

enum class Backend {
    NEON,      // ARM SIMD (default fallback)
    SVE,       // ARM scalable vectors
    APU,       // MediaTek APU
    Vulkan,    // Mali GPU compute
};

const char* backend_name(Backend b);

/// Generate native code for the given backend.
std::string generate(const ir::Graph& graph, Backend backend);

} // namespace forge::codegen
