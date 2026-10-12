#include "forge/core/log.hpp"
#include <spdlog/spdlog.h>
#include "forge/core/error.hpp"
#include "forge/frontend/onnx_parser.hpp"
#include "forge/optimizer/optimizer.hpp"
#include "forge/codegen/codegen.hpp"

#include <CLI/CLI.hpp>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    forge::init_logging();

    CLI::App app{"Forge - AI model compiler for ARM64"};
    app.set_version_flag("--version", "0.1.0");

    std::string input;
    std::string output;
    std::string backend = "neon";

    app.add_option("input", input, "Input model file (.onnx)")
        ->required();
    app.add_option("-o,--output", output, "Output file");
    app.add_option("-b,--backend", backend, "Backend: neon|sve|apu|vulkan")
        ->check(CLI::IsMember({"neon", "sve", "apu", "vulkan"}));

    CLI11_PARSE(app, argc, argv);

    spdlog::info("Forge v0.1.0");
    spdlog::info("Input:   {}", input);
    spdlog::info("Backend: {}", backend);

    try {
        auto graph = forge::frontend::parse_onnx(input);
        spdlog::info("{}", graph.summary());

        auto stats = forge::optimizer::optimize(graph);
        spdlog::info("Optimization: {} -> {} nodes",
                     stats.nodes_before, stats.nodes_after);

        auto backend_enum = forge::codegen::Backend::NEON;
        if      (backend == "sve")    backend_enum = forge::codegen::Backend::SVE;
        else if (backend == "apu")    backend_enum = forge::codegen::Backend::APU;
        else if (backend == "vulkan") backend_enum = forge::codegen::Backend::Vulkan;

        auto code = forge::codegen::generate(graph, backend_enum);
        spdlog::info("Generated {} bytes", code.size());

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
