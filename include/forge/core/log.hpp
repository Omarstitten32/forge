#pragma once

#include <string>

namespace forge {

/// Initialize the global logger with format and level.
void init_logging();

/// Set log level: trace, debug, info, warn, error, critical.
void set_log_level(const std::string& level);

} // namespace forge
