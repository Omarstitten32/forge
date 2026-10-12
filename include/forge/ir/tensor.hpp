#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace forge::ir {

/// Data type of tensor elements.
enum class DataType {
    F32, F16, BF16,
    I8, I16, I32, I64,
    U8,
    Unknown,
};

std::size_t dtype_size(DataType dt);
const char* dtype_name(DataType dt);

/// Multi-dimensional shape.
struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

    std::size_t rank() const { return dims.size(); }
    int64_t     total_elements() const;
    std::string to_string() const;
};

/// Tensor metadata (does not own data).
struct TensorInfo {
    std::string name;
    DataType    dtype{DataType::Unknown};
    Shape       shape;
};

} // namespace forge::ir
