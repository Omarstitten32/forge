#pragma once

#include <stdexcept>
#include <string>

namespace forge {

/// Base exception for all Forge errors.
class ForgeError : public std::runtime_error {
public:
    explicit ForgeError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/// Errors while parsing model files.
class ParseError : public ForgeError {
public:
    explicit ParseError(const std::string& msg)
        : ForgeError("ParseError: " + msg) {}
};

/// Errors during optimization passes.
class OptimizationError : public ForgeError {
public:
    explicit OptimizationError(const std::string& msg)
        : ForgeError("OptimizationError: " + msg) {}
};

/// Errors during code generation.
class CodegenError : public ForgeError {
public:
    explicit CodegenError(const std::string& msg)
        : ForgeError("CodegenError: " + msg) {}
};

/// Errors during runtime execution.
class RuntimeError : public ForgeError {
public:
    explicit RuntimeError(const std::string& msg)
        : ForgeError("RuntimeError: " + msg) {}
};

} // namespace forge
