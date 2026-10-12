#include "forge/ir/tensor.hpp"
#include <sstream>

namespace forge::ir {

std::size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::F32:  return 4;
        case DataType::F16:  return 2;
        case DataType::BF16: return 2;
        case DataType::I8:   return 1;
        case DataType::I16:  return 2;
        case DataType::I32:  return 4;
        case DataType::I64:  return 8;
        case DataType::U8:   return 1;
        case DataType::Unknown: return 0;
    }
    return 0;
}

const char* dtype_name(DataType dt) {
    switch (dt) {
        case DataType::F32:  return "f32";
        case DataType::F16:  return "f16";
        case DataType::BF16: return "bf16";
        case DataType::I8:   return "i8";
        case DataType::I16:  return "i16";
        case DataType::I32:  return "i32";
        case DataType::I64:  return "i64";
        case DataType::U8:   return "u8";
        case DataType::Unknown: return "unknown";
    }
    return "unknown";
}

int64_t Shape::total_elements() const {
    int64_t total = 1;
    for (auto d : dims) total *= d;
    return total;
}

std::string Shape::to_string() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ", ";
        os << dims[i];
    }
    os << "]";
    return os.str();
}

} // namespace forge::ir
