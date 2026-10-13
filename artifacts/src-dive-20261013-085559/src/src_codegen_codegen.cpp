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
    // NOTE: This is a PLACEHOLDER.
    // Real native code generation (NEON/SVE/APU/Vulkan kernels)
    // is Phase 3 work — not yet implemented.
    //
    // Current output is metadata only, intended for:
    //  - Verifying the pipeline (input -> IR -> optimizer -> codegen)
    //  - Format compatibility checks
    //  - Integration testing

    std::string out;
    out += "// =====================================================\n";
    out += "// Forge v0.1.0 — Code Generation STUB\n";
    out += "// =====================================================\n";
    out += "//\n";
    out += "// WARNING: This is NOT real code generation.\n";
    out += "//          Native kernel emission is Phase 3 work.\n";
    out += "//          Current output is metadata only.\n";
    out += "//\n";
    out += "// Target backend : " + std::string(backend_name(backend)) + "\n";
    out += "// Graph summary  : " + graph.summary() + "\n";
    out += "//\n";
    out += "// Planned for Phase 3:\n";
    out += "//   - Node-by-node kernel emission\n";
    out += "//   - SIMD intrinsics (NEON/SVE)\n";
    out += "//   - APU dispatch tables\n";
    out += "//   - Memory layout planning\n";
    out += "// =====================================================\n";
    return out;
}

} // namespace forge::codegen
