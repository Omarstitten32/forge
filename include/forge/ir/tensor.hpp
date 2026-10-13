#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
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

/// A tensor that owns its data (for constants/weights).
/// Uses a raw byte buffer — layout is row-major, tightly packed.
class Tensor {
public:
    Tensor() = default;

    /// Construct from metadata only (no data).
    explicit Tensor(TensorInfo info)
        : info_(std::move(info)) {}

    /// Construct with data (copies bytes).
    Tensor(TensorInfo info, const std::vector<std::uint8_t>& data)
        : info_(std::move(info)), data_(data) {}

    /// Construct with data (moves bytes).
    Tensor(TensorInfo info, std::vector<std::uint8_t>&& data)
        : info_(std::move(info)), data_(std::move(data)) {}

    // ---- Metadata ----
    const TensorInfo& info()  const { return info_; }
    const std::string& name() const { return info_.name; }
    DataType dtype()          const { return info_.dtype; }
    const Shape& shape()      const { return info_.shape; }

    /// Replace metadata (used by shape inference). Does not touch data.
    void set_info(TensorInfo info) { info_ = std::move(info); }

    // ---- Data ----
    bool has_data() const { return !data_.empty(); }
    std::size_t byte_size() const { return data_.size(); }
    std::size_t element_count() const { return info_.shape.total_elements(); }

    const std::uint8_t* data() const { return data_.data(); }
    std::uint8_t*       data()       { return data_.data(); }

    const std::vector<std::uint8_t>& bytes() const { return data_; }
    std::vector<std::uint8_t>&       bytes()       { return data_; }

    // ---- Typed access (no bounds check — caller responsible) ----
    template <typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(data_.data());
    }

    template <typename T>
    T* as() {
        return reinterpret_cast<T*>(data_.data());
    }

    // ---- Summary ----
    std::string summary() const;

private:
    TensorInfo                 info_;
    std::vector<std::uint8_t>  data_;
};

} // namespace forge::ir
