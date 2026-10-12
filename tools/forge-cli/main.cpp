#include "forge/core/log.hpp"
#include "forge/core/error.hpp"
#include "forge/frontend/onnx_parser.hpp"
#include "forge/optimizer/optimizer.hpp"
#include "forge/codegen/codegen.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <iomanip>

int main(int argc, char** argv) {
    forge::init_logging();

    CLI::App app{"Forge - AI model compiler for ARM64"};
    app.set_version_flag("--version", "Forge v0.1.0 (aarch64)");

    std::string input;
    std::string output;
    std::string backend = "neon";

    app.add_option("input", input, "Input model file (.onnx)")->required();
    app.add_option("-o,--output", output, "Output file");
    app.add_option("-b,--backend", backend, "Backend: neon|sve|apu|vulkan")
        ->check(CLI::IsMember({"neon", "sve", "apu", "vulkan"}));

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
