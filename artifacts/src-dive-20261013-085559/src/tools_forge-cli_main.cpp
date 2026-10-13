#include "forge/core/log.hpp"
#include "forge/core/error.hpp"
#include "forge/frontend/onnx_parser.hpp"
#include "forge/optimizer/optimizer.hpp"
#include "forge/codegen/codegen.hpp"

#include <CLI/CLI.hpp>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <iomanip>
#include <sstream>
#include <cstdio>


// ==========================================================
// FNV-1a 64-bit hash (for tensor verification)
// ==========================================================
static std::uint64_t fnv1a_64(const std::uint8_t* data, std::size_t len) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= static_cast<std::uint64_t>(data[i]);
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    forge::init_logging();

    CLI::App app{"Forge - AI model compiler for ARM64"};
    app.set_version_flag("--version", "Forge v0.1.0 (aarch64)");
    app.usage("USAGE:  forge [OPTIONS] <model.onnx>");
    app.footer(
        "EXAMPLES:\n"
        "  forge model.onnx                     Analyze with default backend\n"
        "  forge model.onnx --backend=neon      Use ARM NEON (SIMD)\n"
        "  forge model.onnx --backend=sve       Use ARM SVE/SVE2\n"
        "  forge model.onnx --backend=apu       Use MediaTek APU\n"
        "  forge model.onnx --backend=vulkan    Use Mali GPU\n"
        "  forge model.onnx -o generated.c      Save generated code\n"
        "\n"
        "SUPPORTED FORMATS:\n"
        "  .onnx    ONNX models (Opset 7 through 17+)\n"
        "\n"
        "EXIT CODES:\n"
        "  0    Success\n"
        "  1    Invalid model or parse error\n"
        "  2    Unexpected runtime error\n"
    );

    std::string input;
    std::string output;
    std::string backend = "neon";

    app.add_option("input", input, "Input model file (.onnx)")->required();
    app.add_option("-o,--output", output, "Output file");
    app.add_option("-b,--backend", backend, "Backend: neon|sve|apu|vulkan")
        ->check(CLI::IsMember({"neon", "sve", "apu", "vulkan"}));

    bool verify_tensors = false;
    app.add_flag("--verify-tensors", verify_tensors,
        "Print per-tensor metadata + FNV-1a hash (for validation)");

    CLI11_PARSE(app, argc, argv);

    spdlog::info("Forge v0.1.0");
    spdlog::info("Input:   {}", input);
    spdlog::info("Backend: {}", backend);
    spdlog::info("");

    try {
        auto result = forge::frontend::parse_onnx(input);
        const auto& m = result.metadata;

        // Metadata report
        spdlog::info("═══════ ONNX Metadata ═══════");
        spdlog::info("  Producer   : {} {}", m.producer_name, m.producer_version);
        spdlog::info("  IR version : {}", m.ir_version);
        spdlog::info("  Opset      : {}", m.opset_version);
        spdlog::info("  Nodes      : {}", m.node_count);
        spdlog::info("  Initializers: {}", m.initializer_count);
        spdlog::info("  Inputs     : {}", m.input_count);
        spdlog::info("  Outputs    : {}", m.output_count);
        spdlog::info("");

        spdlog::info("═══════ IR Graph ═══════");
        spdlog::info("  {}", result.graph.summary());

        // ---- Tensor verification mode ----
        if (verify_tensors) {
            std::cout << "=== TENSOR_VERIFY_START ===" << std::endl;
            for (const auto& t : result.graph.tensors()) {
                std::uint64_t h = 0;
                if (t.has_data()) {
                    h = fnv1a_64(t.data(), t.byte_size());
                }
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%016llx",
                              (unsigned long long)h);

                // Format: name | dtype | rank | dims | bytes | hash
                std::ostringstream line;
                line << "TENSOR|" << t.name() << "|"
                     << forge::ir::dtype_name(t.dtype()) << "|";
                for (std::size_t i = 0; i < t.shape().dims.size(); ++i) {
                    if (i) line << ",";
                    line << t.shape().dims[i];
                }
                line << "|" << t.byte_size()
                     << "|" << buf;
                std::cout << line.str() << std::endl;
            }
            std::cout << "=== TENSOR_VERIFY_END ===" << std::endl;
        }

        // Show first few tensors with data
        std::size_t shown = 0;
        for (const auto& t : result.graph.tensors()) {
            if (t.has_data() && shown < 5) {
                spdlog::info("    - {}", t.summary());
                ++shown;
            }
        }
        if (shown == 0) {
            spdlog::info("    (no tensors with data)");
        }
        spdlog::info("");

        auto stats = forge::optimizer::optimize(result.graph);
        spdlog::info("Optimization: {} -> {} nodes",
                     stats.nodes_before, stats.nodes_after);

        auto backend_enum = forge::codegen::Backend::NEON;
        if      (backend == "sve")    backend_enum = forge::codegen::Backend::SVE;
        else if (backend == "apu")    backend_enum = forge::codegen::Backend::APU;
        else if (backend == "vulkan") backend_enum = forge::codegen::Backend::Vulkan;

        auto code = forge::codegen::generate(result.graph, backend_enum);
        spdlog::info("Generated {} bytes of code", code.size());

        if (!output.empty()) {
            std::ofstream out(output);
            if (!out) {
                spdlog::error("Cannot write to {}", output);
                return 1;
            }
            out << code;
            spdlog::info("Written to {}", output);
        }

        return 0;
    } catch (const forge::ForgeError& e) {
        spdlog::error("{}", e.what());
        return 1;
    } catch (const std::exception& e) {
        spdlog::error("Unexpected: {}", e.what());
        return 2;
    }
}
