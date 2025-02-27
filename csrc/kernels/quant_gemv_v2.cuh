// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "dispatch_macros.h"
#include "kernels/convert.cuh"
#include "kernels/copy/mod.cuh"
#include "kernels/quant_gemv_traits.cuh"
#include "kernels/quant_gemv_v2.cuh"
#include "util/common.h"
#include "util/cuda_utils.cuh"

namespace vptq::kernels {

using namespace copy;

template <typename DType, typename IdType, typename ResIdType,
          typename SharedStorage, typename KeTraits>
__global__ void ke_quant_gemv_v2(
    DType* __restrict__ output, const DType* const __restrict__ act,
    const DType* const __restrict__ bias,
    const IdType* const __restrict__ indices,
    const DType* const __restrict__ centroids,
    const DType* const __restrict__ residual_centroids,
    const DType* const __restrict__ scale_weights,
    const DType* const __restrict__ scale_bias,  //
    int64_t batch, int64_t seq_length,           //
    int64_t in_features, int64_t out_features) {
  /// constants
  static constexpr int kTileSize = KeTraits::kTileSize;

  /// === shared memory buffer
  extern __shared__ unsigned char buf_[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(buf_);

  /// === shared memory pointer for output
  DType* s_output = smem.output.data();

  /// === shared memory pointer for intermediate result
  DType* s_quant_weights = smem.dequant_weights.data();

  /// === shared memory pointers for inputs
  DType* s_codebook = smem.codebook.data();
  DType* s_codebook_res = smem.codebook_res.data();
  DType* s_inputs = smem.inputs.data();
  IdType* s_ids = smem.indices.data();
  ResIdType* s_res_ids = reinterpret_cast<ResIdType*>(s_ids + kTileSize);
  DType* s_scale_weights = scale_weights ? s_inputs + kTileSize : nullptr;
  DType* s_scale_bias = scale_bias ? s_scale_weights + kTileSize : nullptr;
  DType* s_bias = bias ? s_output + KeTraits::kVecLen : nullptr;

  /// === 1. load data from global to shared memory === ///
  typename KeTraits::MainCentroidTraits::Loader loader;
  // load the main centroids into shared memory
  loader(centroids, s_codebook);
  if (residual_centroids) {
    // load the residual centroids into shared memory if available
    typename KeTraits::ResCentroidTraits::Loader loader;
    loader(residual_centroids, s_codebook_res);
  }
  __copy_async();
  __syncthreads();

  if (bias && threadIdx.x < KeTraits::kBiasLoadThreads) {
    // load the bias if avaliable.
    int thread_offset = threadIdx.x * KeTraits::kNumPerAccess;
    const DType* src_ptr =
        bias + blockIdx.z * KeTraits::kVecLen + thread_offset;

    ld_global_st_shared<16>(  // a single thread access 16 bytes data
        static_cast<uint32_t>(__cvta_generic_to_shared(s_bias + thread_offset)),
        src_ptr);
  }

  // advance the input pointer to the current CTA
  int batch_id = blockIdx.x / seq_length;
  int seq_id = blockIdx.x % seq_length;
  const DType* x =
      act + batch_id * (seq_length * in_features) + seq_id * in_features;

  typename KeTraits::InputLoader input_loader;
  typename KeTraits::IndexLoader index_loader;
  typename KeTraits::WarpCounter counter;

  int wid = warpid();
  for (int step = 0; step < in_features; step += kTileSize) {  // over tiles
    counter.reset();
    if (wid < counter.next()) {
      // load a single tile of the activation into shared memory
      input_loader(x + step, s_inputs, counter.cur());
    }

    ++counter;
    if (wid >= counter.cur() && wid < counter.next()) {
      // load indices into shared memory
      index_loader(indices + step, s_ids, counter.cur());
    }

    if (scale_weights) {
      ++counter;
      if (wid >= counter.cur() && wid < counter.next()) {
        // load scale_weights into shared memory if available
        input_loader(scale_weights + step, s_scale_weights, counter.cur());
      }

      ++counter;
      if (wid >= counter.cur() && wid < counter.next()) {
        // load scale_bias into shared memory if available
        input_loader(scale_bias + step, s_scale_bias, counter.cur());
      }
    }
    __copy_async();
    __syncthreads();

    /// === 2. decode, add residual, and scale on register === ///
    typename KeTraits::Decoder decoder;
    decoder(s_quant_weights, s_codebook, s_codebook_res, s_ids, s_res_ids,
            s_scale_weights, s_scale_bias);

    /// === 3. dot product and apply bias === ///

    /// === 4. accumulate results between tiles and store === ///
  }

  return;
}

}  // namespace vptq::kernels
