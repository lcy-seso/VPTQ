// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "config.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace vptq {

DEVICE int warpid() { return threadIdx.x / WARP_SIZE; }

DEVICE int laneid() { return threadIdx.x % WARP_SIZE; }

template <typename T>
struct C10ToNvType;

template <>
struct C10ToNvType<c10::BFloat16> {
  typedef __bfloat16 type;
};

template <>
struct C10ToNvType<c10::Half> {
  typedef __half type;
};

template <>
struct C10ToNvType<float> {
  typedef float type;
};

namespace cuda {

constexpr int kBlockSize = 256;

template <typename T>
struct TypeVec2 {
  typedef __half2 type;
};

template <>
struct TypeVec2<__bfloat16> {
  typedef __bfloat162 type;
};

template <>
struct TypeVec2<float> {
  typedef float2 type;
};

template <unsigned int WarpSize>
DEVICE float warpReduceSum(float sum) {
  if constexpr (WarpSize >= 64)
    sum += SHFL_DOWN(sum, 32);  // 0-16, 1-17, 2-18, etc.
  if constexpr (WarpSize >= 32)
    sum += SHFL_DOWN(sum, 16);  // 0-16, 1-17, 2-18, etc.
  if constexpr (WarpSize >= 16)
    sum += SHFL_DOWN(sum, 8);  // 0-8, 1-9, 2-10, etc.
  if constexpr (WarpSize >= 8) sum += SHFL_DOWN(sum, 4);  // 0-4, 1-5, 2-6, etc.
  if constexpr (WarpSize >= 4)
    sum += SHFL_DOWN(sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if constexpr (WarpSize >= 2) sum += SHFL_DOWN(sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <int GROUPSIZE, typename T>
DEVICE void ldg_vec_x(T* __restrict__ dst_t32,
                      const uint32_t* __restrict__ src_u32) {
  uint32_t* dst_u32 = (uint32_t*)dst_t32;
  if constexpr (std::is_same<T, float>::value ||
                std::is_same<T, float2>::value) {
    return ldg_vec_x<GROUPSIZE * 2>(dst_u32, src_u32);
  }
  int2* dst = (int2*)dst_u32;
  const int2* src = (const int2*)src_u32;
  if constexpr (GROUPSIZE == 2) {
    *dst_u32 = VPTQ_LDG(src_u32);
  } else if constexpr (GROUPSIZE == 4) {
    *dst = VPTQ_LDG(src);
  } else if constexpr (GROUPSIZE == 6) {
    dst_u32[0] = VPTQ_LDG(src_u32);
    dst_u32[1] = VPTQ_LDG(src_u32 + 1);
    dst_u32[2] = VPTQ_LDG(src_u32 + 2);
  } else if constexpr (GROUPSIZE == 8) {
    *(int4*)dst = VPTQ_LDG((const int4*)src);
  } else if constexpr (GROUPSIZE == 16) {
    *(int4*)dst = VPTQ_LDG((const int4*)src);
    *(int4*)(dst + 2) = VPTQ_LDG((const int4*)(src + 2));
  } else if constexpr (GROUPSIZE == 12) {
    if (uint64_t(src) % 16) {
      dst[0] = VPTQ_LDG(src);
      int4 b = VPTQ_LDG((const int4*)(src + 1));
      dst[1] = *((const int2*)&b);
      dst[2] = *((const int2*)&b + 1);
    } else {
      *(int4*)dst = VPTQ_LDG((int4*)(src));
      dst[2] = VPTQ_LDG((src + 2));
    }
  } else if constexpr (GROUPSIZE == 24) {
    *((int4*)(dst)) = VPTQ_LDG((const int4*)(src));
    *(((int4*)(dst)) + 1) = VPTQ_LDG(((const int4*)(src)) + 1);
    *(((int4*)(dst)) + 2) = VPTQ_LDG(((const int4*)(src)) + 2);
  } else if constexpr (GROUPSIZE == 32) {
    *((int4*)(dst)) = VPTQ_LDG((const int4*)(src));
    *(((int4*)(dst)) + 1) = VPTQ_LDG(((const int4*)(src)) + 1);
    *(((int4*)(dst)) + 2) = VPTQ_LDG(((const int4*)(src)) + 2);
    *(((int4*)(dst)) + 3) = VPTQ_LDG(((const int4*)(src)) + 3);
  } else {
    assert(false);
  }
}

template <int WBITS>
DEVICE uint32_t iterator_packed_tensor(const uint32_t* ptr, int idx) {
  if constexpr (WBITS == 32) {
    return ptr[idx];
  } else if constexpr (WBITS == 16) {
    return ((const uint16_t*)ptr)[idx];
  } else {
    int start_bits = idx * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;
    uint32_t v = (ptr[first] >> (start_bits)) & (uint32_t(1 << WBITS) - 1);
    if (first == second || end_bits == 0) {
      return v;
    } else {
      // second position might be out of bound
      uint32_t sec_v = ptr[second];
      v |= ((sec_v) & ((1 << (end_bits)) - 1)) << (32 - start_bits);
      return v;
    }
  }
}
}  // namespace cuda

template <typename T>
DEVICE T FMA2(T a, T b, T c) {
  if constexpr (std::is_same<T, __bfloat162>::value) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    float x =
        __bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(c.x);
    float y =
        __bfloat162float(a.y) * __bfloat162float(b.y) + __bfloat162float(c.y);
    return __bfloat162{__float2bfloat16(x), __float2bfloat16(y)};
#else
    return __hfma2(a, b, c);
#endif
  } else if constexpr (std::is_same<T, float2>::value) {
    return float2{a.x * b.x + c.x, a.y * b.y + c.y};
  } else {
    return __hfma2(a, b, c);
  }
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename T>
DEVICE T FMA(T a, T b, T c) {
  if constexpr (std::is_same<T, __bfloat16>::value) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
    float x = __bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c);
    return __bfloat16{__float2bfloat16(x)};
#else
    return __hfma(a, b, c);
#endif
  } else if constexpr (std::is_same<T, float>::value) {
    return float{a.x * b.x + c.x};
  } else {
    return __hfma(a, b, c);
  }
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename T>
DEVICE T ADD2(T a, T b) {
  if constexpr (std::is_same<T, __bfloat162>::value) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(USE_ROCM)
    float x = __bfloat162float(a.x) + __bfloat162float(b.x);
    float y = __bfloat162float(a.y) + __bfloat162float(b.y);
    return __bfloat162{__float2bfloat16(x), __float2bfloat16(y)};
#else
    return __hadd2(a, b);
#endif
  } else if constexpr (std::is_same<T, float2>::value) {
    return float2{a.x + b.x, a.y + b.y};
  } else {
    return __hadd2(a, b);
  }
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename T>
DEVICE T ZERO_VALUE(T a) {
  if constexpr (std::is_same<T, __bfloat16>::value) {
#if defined(USE_ROCM)
    return __float2bfloat16(0.0f);
#else
    return __float2bfloat16_rn(0.0f);
#endif
  } else if constexpr (std::is_same<T, float>::value) {
    return 0.0f;
  } else {
    return __float2half(0.0f);
  }
}

#if defined(USE_ROCM)
DEVICE __half operator+(const __half& a, const __half& b) {
  // Use HIP's intrinsic __hadd for half-precision addition
  return __hadd(a, b);
}

// Overload the * operator for __half
DEVICE __half operator*(const __half& a, const __half& b) {
  // Use HIP's intrinsic __hmul for half-precision multiplication
  return __hmul(a, b);
}
#endif

}  // namespace vptq
