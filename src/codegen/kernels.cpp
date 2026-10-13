#include "forge/codegen/kernels.hpp"

#include <cmath>
#include <cstring>

namespace forge::codegen {

// ==========================================================
//  Conv2D (naive)
// ==========================================================
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
    std::size_t H_out, std::size_t W_out)
{
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t co = 0; co < C_out; ++co) {
            const float b = bias ? bias[co] : 0.0f;

            for (std::size_t oh = 0; oh < H_out; ++oh) {
                for (std::size_t ow = 0; ow < W_out; ++ow) {
                    float sum = b;

                    for (std::size_t ci = 0; ci < C_in; ++ci) {
                        for (std::size_t kh = 0; kh < KH; ++kh) {
                            for (std::size_t kw = 0; kw < KW; ++kw) {
                                const std::size_t ih = oh * stride_h + kh;
                                const std::size_t iw = ow * stride_w + kw;

                                // Padding check
                                if (ih < pad_top || iw < pad_left) continue;
                                const std::size_t y = ih - pad_top;
                                const std::size_t x = iw - pad_left;
                                if (y >= H || x >= W) continue;

                                const std::size_t in_idx =
                                    ((n * C_in + ci) * H + y) * W + x;
                                const std::size_t w_idx =
                                    ((co * C_in + ci) * KH + kh) * KW + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }

                    const std::size_t out_idx =
                        ((n * C_out + co) * H_out + oh) * W_out + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// ==========================================================
//  Depthwise Conv2D
// ==========================================================
void conv2d_depthwise_f32(
    const float* input, const float* weight, const float* bias, float* output,
    std::size_t N, std::size_t C, std::size_t H, std::size_t W,
    std::size_t KH, std::size_t KW,
    std::size_t stride_h, std::size_t stride_w,
    std::size_t pad_top, std::size_t pad_left,
    std::size_t H_out, std::size_t W_out)
{
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < C; ++c) {
            const float b = bias ? bias[c] : 0.0f;
            const float* wc = weight + c * KH * KW;
            for (std::size_t oh = 0; oh < H_out; ++oh) {
                for (std::size_t ow = 0; ow < W_out; ++ow) {
                    float sum = b;
                    for (std::size_t kh = 0; kh < KH; ++kh) {
                        for (std::size_t kw = 0; kw < KW; ++kw) {
                            const std::size_t ih = oh * stride_h + kh;
                            const std::size_t iw = ow * stride_w + kw;
                            if (ih < pad_top || iw < pad_left) continue;
                            const std::size_t y = ih - pad_top;
                            const std::size_t x = iw - pad_left;
                            if (y >= H || x >= W) continue;
                            const std::size_t in_idx = ((n * C + c) * H + y) * W + x;
                            sum += input[in_idx] * wc[kh * KW + kw];
                        }
                    }
                    const std::size_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// ==========================================================
//  Gemm (naive)
// ==========================================================
void gemm_f32_naive(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    std::size_t M, std::size_t K, std::size_t N,
    float alpha, float beta,
    bool transA, bool transB)
{
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                const float a = transA
                    ? A[k * M + i]
                    : A[i * K + k];
                const float b = transB
                    ? B[j * K + k]
                    : B[k * N + j];
                sum += a * b;
            }

            const float bias_val = bias ? bias[j] : 0.0f;
            const float c_val    = C[i * N + j];

            C[i * N + j] = alpha * sum + beta * c_val + bias_val;
        }
    }
}

// ==========================================================
//  Element-wise ops
// ==========================================================
void relu_f32(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

void sigmoid_f32(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void tanh_f32(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

void gelu_f32(const float* in, float* out, std::size_t n) {
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float x3 = x * x * x;
        const float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void add_f32(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void mul_f32(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// ==========================================================
//  BatchNorm (inference)
// ==========================================================
void batchnorm_f32(
    const float* in,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    float* out,
    std::size_t N, std::size_t C,
    std::size_t H, std::size_t W,
    float epsilon)
{
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < C; ++c) {
            const float inv_std = 1.0f / std::sqrt(var[c] + epsilon);
            const float scale   = gamma[c] * inv_std;
            const float shift   = beta[c] - mean[c] * scale;

            for (std::size_t h = 0; h < H; ++h) {
                for (std::size_t w = 0; w < W; ++w) {
                    const std::size_t idx =
                        ((n * C + c) * H + h) * W + w;
                    out[idx] = in[idx] * scale + shift;
                }
            }
        }
    }
}

// ==========================================================
//  MaxPool 2D
// ==========================================================
void maxpool2d_f32(
    const float* in,
    float* out,
    std::size_t N, std::size_t C,
    std::size_t H, std::size_t W,
    std::size_t KH, std::size_t KW,
    std::size_t stride_h, std::size_t stride_w,
    std::size_t H_out, std::size_t W_out)
{
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t oh = 0; oh < H_out; ++oh) {
                for (std::size_t ow = 0; ow < W_out; ++ow) {
                    float best = -INFINITY;

                    for (std::size_t kh = 0; kh < KH; ++kh) {
                        for (std::size_t kw = 0; kw < KW; ++kw) {
                            const std::size_t ih = oh * stride_h + kh;
                            const std::size_t iw = ow * stride_w + kw;
                            if (ih >= H || iw >= W) continue;

                            const std::size_t idx =
                                ((n * C + c) * H + ih) * W + iw;
                            if (in[idx] > best) best = in[idx];
                        }
                    }

                    const std::size_t out_idx =
                        ((n * C + c) * H_out + oh) * W_out + ow;
                    out[out_idx] = best;
                }
            }
        }
    }
}

  // ==========================================================
  //  Global Average Pool
  // ==========================================================
  void globalavgpool_f32(
      const float* in,
      float* out,
      std::size_t N, std::size_t C,
      std::size_t H, std::size_t W)
  {
      const float inv = 1.0f / static_cast<float>(H * W);
      for (std::size_t n = 0; n < N; ++n) {
          for (std::size_t c = 0; c < C; ++c) {
              float sum = 0.0f;
              const std::size_t base = (n * C + c) * H * W;
              for (std::size_t i = 0; i < H * W; ++i) {
                  sum += in[base + i];
              }
              out[n * C + c] = sum * inv;
          }
      }
  }

// ==========================================================
//  Flatten (NCHW -> [N, C*H*W]) — no reorder
// ==========================================================
void flatten_f32(const float* in, float* out, std::size_t n) {
    std::memcpy(out, in, n * sizeof(float));
}

} // namespace forge::codegen
