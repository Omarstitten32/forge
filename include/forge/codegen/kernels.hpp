#pragma once

#include <cstddef>
#include <cstdint>

namespace forge::codegen {

// ==========================================================
//  Naive C++ kernels for Phase 3a
//  These are portable reference implementations.
//  They prioritize correctness over speed.
// ==========================================================

/// Conv2D — NCHW layout, F32.
///
/// input   : [N, C_in, H, W]
/// weight  : [C_out, C_in, KH, KW]
/// bias    : [C_out] or nullptr
/// output  : [N, C_out, H_out, W_out]
void conv2d_f32_naive(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::size_t N,
    std::size_t C_in, std::size_t H, std::size_t W,
    std::size_t C_out,
    std::size_t KH, std::size_t KW,
    std::size_t stride_h, std::size_t stride_w,
    std::size_t pad_top,  std::size_t pad_left,
    std::size_t H_out, std::size_t W_out);

/// Depthwise Conv2D — NCHW, F32. One filter per input channel.
/// weight: [C, 1, KH, KW]
void conv2d_depthwise_f32(
    const float* input, const float* weight, const float* bias, float* output,
    std::size_t N, std::size_t C, std::size_t H, std::size_t W,
    std::size_t KH, std::size_t KW,
    std::size_t stride_h, std::size_t stride_w,
    std::size_t pad_top, std::size_t pad_left,
    std::size_t H_out, std::size_t W_out);

/// Gemm — C = alpha * A * B + beta * C (or bias).
/// A: [M, K] (row-major)
/// B: [K, N] (row-major, unless transB)
/// C: [M, N]
void gemm_f32_naive(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    std::size_t M, std::size_t K, std::size_t N,
    float alpha, float beta,
    bool transA, bool transB);

/// Element-wise Relu.
void relu_f32(const float* in, float* out, std::size_t n);

/// Element-wise Sigmoid.
void sigmoid_f32(const float* in, float* out, std::size_t n);

/// Element-wise Tanh.
void tanh_f32(const float* in, float* out, std::size_t n);

/// Element-wise Gelu (tanh approximation).
void gelu_f32(const float* in, float* out, std::size_t n);

/// Element-wise Add.
void add_f32(const float* a, const float* b, float* out, std::size_t n);

/// Element-wise Mul.
void mul_f32(const float* a, const float* b, float* out, std::size_t n);

/// BatchNormalization (inference mode).
/// x_norm = (x - mean) / sqrt(var + eps) * gamma + beta
void batchnorm_f32(
    const float* in,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    float* out,
    std::size_t N, std::size_t C,
    std::size_t H, std::size_t W,
    float epsilon);

/// MaxPool 2D.
void maxpool2d_f32(
    const float* in,
    float* out,
    std::size_t N, std::size_t C,
    std::size_t H, std::size_t W,
    std::size_t KH, std::size_t KW,
    std::size_t stride_h, std::size_t stride_w,
    std::size_t H_out, std::size_t W_out);

/// Global Average Pool — NCHW -> [N, C, 1, 1].
void globalavgpool_f32(
    const float* in, float* out,
    std::size_t N, std::size_t C,
    std::size_t H, std::size_t W);

/// Flatten — copy with no reordering (NCHW -> [N, C*H*W]).
void flatten_f32(const float* in, float* out, std::size_t n);

} // namespace forge::codegen
