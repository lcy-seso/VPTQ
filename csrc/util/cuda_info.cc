// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "util/cuda_info.hpp"

namespace vptq {

int GetMaxSmemPerBlock(int device_id) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  return deviceProp.sharedMemPerBlock;
}

int GetSharedMemPerSM(int device_id) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  return deviceProp.sharedMemPerMultiprocessor;
}

}  // namespace vptq
