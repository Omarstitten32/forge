#include "forge/core/log.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace forge {

void init_logging() {
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger  = std::make_shared<spdlog::logger>("forge", console);

    logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);
}

void set_log_level(const std::string& level) {
    auto lvl = spdlog::level::from_str(level);
    spdlog::set_level(lvl);
}

} // namespace forge
