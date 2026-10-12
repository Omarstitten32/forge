#pragma once

#include "forge/ir/graph.hpp"
#include <memory>

namespace forge::runtime {

/// Execution context for a compiled graph.
class Context {
public:
    Context();
    ~Context();

    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&)                 = default;
    Context& operator=(Context&&)      = default;

    /// Total number of nodes in the associated graph.
    std::size_t node_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace forge::runtime
