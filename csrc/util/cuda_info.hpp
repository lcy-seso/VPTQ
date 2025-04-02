// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>

namespace vptq {

int GetMaxSmemPerBlock(int device_id);

int GetSharedMemPerSM(int device_id);

}  // namespace vptq
