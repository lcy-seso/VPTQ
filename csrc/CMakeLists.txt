# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------
set(TARGET "vptq")
file(GLOB_RECURSE SOURCES "*.cu" "*.cc")
message(STATUS "Building ${TARGET} with ${SOURCES}")

cuda_add_library(${TARGET} SHARED ${SOURCES})

set_target_properties(
  ${TARGET}
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             CUDA_EXTENSIONS OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_SEPARABLE_COMPILATION ON)

# Refer to this issue for more context:
# https://github.com/pytorch/pytorch/issues/13541
target_compile_definitions(${TARGET} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

# The find_package(Torch) command does not expose PyTorch's Python bindings.
# However, when using Pybind11, we need to link against these bindings.
list(APPEND TORCH_LIBRARIES "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

target_compile_options(
  ${TARGET}
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
         -Werror,-Wall
         -rdc=true
         -std=c++17
         -fconcepts
         -fpermissive
         --use_fast_math
         --generate-line-info>)
target_compile_features(${TARGET} PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")
