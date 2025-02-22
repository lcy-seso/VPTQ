// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "util/debug.cuh"

namespace vptq::kernels {

template <typename DType_, typename IdType_, typename ResIdType_,
          const int kNumPerThread_, const int kVecLen_>
struct WeightDecoder {
  using DType = DType_;
  using IdType = IdType_;
  using ResIdType = ResIdType_;

  static constexpr int kNumPerThread = kNumPerThread_;
  static constexpr int kVecLen = kVecLen_;

  DEVICE void operator()(DType* output,  // output
                         const DType* codebook,
                         const DType* codebook_res,  // codebooks
                         const IdType* ids,
                         const ResIdType* res_ids,  // indices
                         const DType* alpha, const DType* beta) {
    int tid = threadIdx.x;  // threads in a CTA are laid out in 1-D fashion.

    // data tile on registers
    IdType ids_[kNumPerThread];
    ResIdType res_ids_[kNumPerThread];

    uint32_t* ids_ptr = reinterpret_cast<uint32_t*>(ids_);
    uint32_t* res_ids_ptr = reinterpret_cast<uint32_t*>(res_ids_);

    DType weights_[kNumPerThread * kVecLen];
    DType residual_weights_[kNumPerThread * kVecLen];

#pragma unroll
    for (int i = 0; i < kNumPerThread; ++i) {
    }

    // if (thread(0)) {
    //   for (int i = 0; i < 512; ++i) {
    //     printf("%u, ", (unsigned int)i);
    //     if (i && (i + 1) % 16 == 0) printf("\n");
    //   }
    // }
  }
};

}  // namespace vptq::kernels
