// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define DISPATCH_TYPE_CASE(TYPE, NV_TYPE, ...) \
  case TYPE: {                                 \
    using nv_type = NV_TYPE;                   \
    return __VA_ARGS__();                      \
  }

#define VPTQ_DISPATCH_TYPES(TYPE, ...)                                       \
  c10::ScalarType _type = TYPE;                                              \
  [&] {                                                                      \
    switch (_type) {                                                         \
      DISPATCH_TYPE_CASE(c10::ScalarType::Half, __half, __VA_ARGS__)         \
      DISPATCH_TYPE_CASE(c10::ScalarType::BFloat16, __bfloat16, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR("Dispatch is not implemented for type: '", toString(_type), \
                 "'");                                                       \
    }                                                                        \
  }();

#define VPTQ_DISPATCH_VEC_LENGTH(VEC_LEN, ...)                                \
  [&] {                                                                       \
    switch (VEC_LEN) {                                                        \
      case 4: {                                                               \
        static constexpr int kVecLen = 4;                                     \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 8: {                                                               \
        static constexpr int kVecLen = 8;                                     \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 12: {                                                              \
        static constexpr int kVecLen = 12;                                    \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 16: {                                                              \
        static constexpr int kVecLen = 16;                                    \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        AT_ERROR("Dispatch is not implemented for vector length: ", VEC_LEN); \
    }                                                                         \
  }();

#define VPTQ_DISPATCH_NUM_CENTROIDS(NUM_CENTROIDS, ...)                \
  [&] {                                                                \
    switch (NUM_CENTROIDS) {                                           \
      case 8192: {                                                     \
        static constexpr int kNumCentroids = 8192;                     \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        AT_ERROR("Dispatch is not implemented for centroids number: ", \
                 NUM_CENTROIDS);                                       \
    }                                                                  \
  }();
