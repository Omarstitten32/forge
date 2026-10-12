#pragma once

#include "forge/ir/graph.hpp"
#include <filesystem>

namespace forge::frontend {

/// Parse an ONNX model file into an IR Graph.
/// Throws ParseError if the file is invalid or unsupported.
ir::Graph parse_onnx(const std::filesystem::path& path);

} // namespace forge::frontend
