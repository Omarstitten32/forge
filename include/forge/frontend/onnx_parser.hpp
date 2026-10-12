#pragma once

#include "forge/ir/graph.hpp"
#include <filesystem>
#include <optional>
#include <string>

namespace forge::frontend {

/// Metadata extracted from an ONNX ModelProto.
struct OnnxMetadata {
    std::string producer_name;
    std::string producer_version;
    std::string domain;
    std::int64_t model_version{0};
    std::int64_t ir_version{0};
    std::string doc_string;
    std::size_t opset_version{0};
    std::size_t node_count{0};
    std::size_t initializer_count{0};
    std::size_t input_count{0};
    std::size_t output_count{0};
};

/// Result of parsing an ONNX file.
struct OnnxParseResult {
    ir::Graph   graph;
    OnnxMetadata metadata;
};

/// Parse an ONNX model file (.onnx, binary protobuf format).
/// Throws ParseError on any failure.
OnnxParseResult parse_onnx(const std::filesystem::path& path);

/// Parse an ONNX model from an in-memory byte buffer.
OnnxParseResult parse_onnx_bytes(const std::vector<std::uint8_t>& data);

} // namespace forge::frontend
