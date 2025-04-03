// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dispatch_macros.h"
#include "kernels/quant_gemv_traits.cuh"
#include "kernels/quant_gemv_v2.cuh"
#include "util/common.h"

namespace vptq {

/**
 * @brief Quantized GEMV kernel.
 * @param act The input activations.
 * @param bias The bias.
 * @param indices The indices.
 * @param centroids The codebook for the main vector quantized weights.
 *        Stored in row-major order. Element type: fp16, bf16.
 *        Shape: (num_codebooks, num_centroids, vec_len).
 * @param residual_centroids The residual centroids.
 * @param scale_weights The scale weights.
 * @param scale_bias The scale bias.
 * @param in_features The number of input features.
 * @param out_features The number of output features.
 */
torch::Tensor quant_gemv_v2(
    const torch::Tensor& act,
    const c10::optional<torch::Tensor>& bias,  //
    const torch::Tensor& indices,              //
    const torch::Tensor& centroids,            //
    const c10::optional<torch::Tensor>& residual_indices,
    const c10::optional<torch::Tensor>& scale_weights,
    const c10::optional<torch::Tensor>& scale_bias,  //
    int64_t num_res_centroids, int64_t out_features) {
  CHECK_INPUT(act);
  CHECK_INPUT(indices);
  CHECK_INPUT(centroids);

  const at::ScalarType dtype = act.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "the activations must be either half-precision (fp16) or bfloat16.");
  TORCH_CHECK(
      centroids.scalar_type() == dtype,
      "the main centroids must be either half-precision (fp16) or bfloat16.");

  TORCH_CHECK_EQ(act.ndimension(), 3);
  TORCH_CHECK_EQ(centroids.ndimension(), 3);

  const int64_t batch = act.size(0);
  const int64_t seq_length = act.size(1);
  const int64_t in_features = act.size(2);

  const int64_t num_codebooks = centroids.size(0);
  const int64_t total_centroids = centroids.size(1);
  const int64_t num_centroids = total_centroids - num_res_centroids;
  const int64_t vec_len = centroids.size(2);

  TORCH_CHECK_LT(batch * seq_length, 16)
      << "In GEMV, the batch size is suggested to be less than 16.";
  TORCH_CHECK_EQ(num_codebooks, 1) << "Only support one codebook.";
  TORCH_CHECK(
      vec_len == 4 || vec_len == 8 || vec_len == 16,
      "Supported vector length in vectorized quantization: 4, 8, or 16.");

  if (scale_weights.has_value()) {
    CHECK_INPUT(scale_weights.value());
    CHECK_INPUT(scale_bias.value());

    TORCH_CHECK(
        scale_weights.value().scalar_type() == dtype,
        "the scale weights must be either half-precision (fp16) or bfloat16.");
    TORCH_CHECK(
        scale_bias.value().scalar_type() == dtype,
        "the scale bias must be either half-precision (fp16) or bfloat16.");
  }

  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
    TORCH_CHECK(bias.value().scalar_type() == dtype,
                "the bias must be either half-precision (fp16) or bfloat16.");
  }

  torch::Tensor output;
  output = at::empty({batch, seq_length, out_features}, centroids.options());

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int block_z = divup<int64_t, int64_t, int64_t>(out_features, vec_len);
  dim3 blocks(batch * seq_length, num_codebooks, block_z);

  const int kMaxSharedMemPerBlock = 48 * 1024;

  VPTQ_DISPATCH_TYPES(dtype, [&] {
    VPTQ_DISPATCH_VEC_LENGTH(vec_len, [&] {
      VPTQ_DISPATCH_NUM_CENTROIDS(num_centroids, [&] {
        VPTQ_DISPATCH_RES_NUM_CENTROIDS(num_res_centroids, [&] {
          const ResIdType* residual_indices_ptr =
              residual_indices.has_value()
                  ? reinterpret_cast<const ResIdType*>(
                        residual_indices.value().data_ptr())
                  : nullptr;

          const DType* bias_ptr =
              bias.has_value()
                  ? reinterpret_cast<const DType*>(bias.value().data_ptr())
                  : nullptr;

          const DType* scale_weights_ptr =
              scale_weights.has_value() ? reinterpret_cast<const DType*>(
                                              scale_weights.value().data_ptr())
                                        : nullptr;

          const DType* scale_bias_ptr = scale_bias.has_value()
                                            ? reinterpret_cast<const DType*>(
                                                  scale_bias.value().data_ptr())
                                            : nullptr;

          // NOTE: IdType and ResIdType are declared in the dispatch macros.
          using Config =
              kernels::QuantGemvKeTraits<DType, IdType, ResIdType, kVecLen,
                                         kNumCentroids, kNumResCentroids>;
          using SharedStorage = Config::SharedStorage;

          auto kernel = &kernels::ke_quant_gemv_v2<DType, IdType, ResIdType,
                                                   SharedStorage, Config>;

          static constexpr int smem_size = SharedStorage::kSmemSize;
          if (smem_size > kMaxSharedMemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
          }

          dim3 threads(Config::kThreads, 1, 1);

          kernel<<<blocks, threads, smem_size, stream>>>(
              reinterpret_cast<DType*>(output.mutable_data_ptr()),
              reinterpret_cast<const DType*>(act.data_ptr()), bias_ptr,
              indices.data_ptr<IdType>(),
              reinterpret_cast<const DType*>(centroids.data_ptr()),
              residual_indices_ptr, scale_weights_ptr, scale_bias_ptr, batch,
              seq_length, in_features, out_features);
        });
      });
    });
  });

  return output;
}
}  // namespace vptq
