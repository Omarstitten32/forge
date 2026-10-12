#include "forge/codegen/codegen.hpp"

namespace forge::codegen {

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::NEON:   return "neon";
        case Backend::SVE:    return "sve";
        case Backend::APU:    return "apu";
        case Backend::Vulkan: return "vulkan";
    }
    return "unknown";
}

std::string generate(const ir::Graph& graph, Backend backend) {
    // TODO(Phase 3): emit NEON/SVE/APU/Vulkan kernels
    std::string out;
    out += "// Forge generated code\n";
    out += "// Backend: ";
    out += backend_name(backend);
    out += "\n";
    out += "// Graph summary: ";
    out += graph.summary();
    out += "\n";
    return out;
}

} // namespace forge::codegen
