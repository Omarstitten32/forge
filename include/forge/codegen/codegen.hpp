#pragma once
#include "forge/ir/graph.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace forge::codegen {

enum class Backend {
    NEON, SVE, APU, Vulkan,
};

const char* backend_name(Backend b);

struct GeneratedCode {
    std::string source;                    // C++ source
    std::vector<std::uint8_t> weights;     // raw f32 LE weights, concatenated
    std::string signature;                 // human-readable summary
};

/// Generate native code + separate weight blob.
GeneratedCode generate(const ir::Graph& graph, Backend backend);

} // namespace forge::codegen
