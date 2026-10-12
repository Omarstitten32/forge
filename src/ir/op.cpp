#include "forge/ir/op.hpp"

namespace forge::ir {

const char* op_name(OpType op) {
    switch (op) {
        case OpType::Add:       return "Add";
        case OpType::Sub:       return "Sub";
        case OpType::Mul:       return "Mul";
        case OpType::Div:       return "Div";
        case OpType::MatMul:    return "MatMul";
        case OpType::Gemm:      return "Gemm";
        case OpType::Conv2D:    return "Conv2D";
        case OpType::Relu:      return "Relu";
        case OpType::Sigmoid:   return "Sigmoid";
        case OpType::Tanh:      return "Tanh";
        case OpType::Softmax:   return "Softmax";
        case OpType::Gelu:      return "Gelu";
        case OpType::BatchNorm: return "BatchNorm";
        case OpType::LayerNorm: return "LayerNorm";
        case OpType::MaxPool:   return "MaxPool";
        case OpType::AvgPool:   return "AvgPool";
        case OpType::Reshape:   return "Reshape";
        case OpType::Transpose: return "Transpose";
        case OpType::Concat:    return "Concat";
        case OpType::Constant:  return "Constant";
        case OpType::Unknown:   return "Unknown";

        // Fused
        case OpType::Conv2DRelu:        return "Conv2DRelu";
        case OpType::Conv2DBatchNorm:   return "Conv2DBatchNorm";
        case OpType::Conv2DBatchNormRelu:return "Conv2DBatchNormRelu";
        case OpType::AddRelu:           return "AddRelu";
        case OpType::MulAdd:            return "MulAdd";
    }
    return "Unknown";
}

bool is_fused(OpType op) {
    switch (op) {
        case OpType::Conv2DRelu:
        case OpType::Conv2DBatchNorm:
        case OpType::Conv2DBatchNormRelu:
        case OpType::AddRelu:
        case OpType::MulAdd:
            return true;
        default:
            return false;
    }
}

} // namespace forge::ir
