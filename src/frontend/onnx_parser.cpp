#include "forge/frontend/onnx_parser.hpp"
#include "forge/core/error.hpp"

namespace forge::frontend {

ir::Graph parse_onnx(const std::filesystem::path& path) {
    // TODO(Phase 1): Implement protobuf-based ONNX parsing
    throw ParseError("ONNX parser not yet implemented: " + path.string());
}

} // namespace forge::frontend
