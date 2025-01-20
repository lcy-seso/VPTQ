#!/bin/bash

# rm -rf dist
# python3 setup.py build bdist_wheel

# python3 setup.py clean

# export TORCH_CUDA_ARCH_LIST="7.5"
# export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
# python3 setup.py build bdist_wheel 2>&1 | tee build.log

# unset TORCH_CUDA_ARCH_LIST
# export NVCC_THREADS=`nproc`

# python3 setup.py clean
# python3 setup.py develop 2>&1 | tee build.log

# TORCH_VERSION=`python -c "import torch; print(torch.__version__)"`
# python3 setup.py build bdist_wheel \
#   --dist-dir="../torch$TORCH_VERSION-dist" 2>&1 | tee build.log

cd build/temp.linux-x86_64-cpython-312

# if [ -d "CMakeFiles" ]; then
#   rm -rf CMakeFiles
# fi

# if [ -f "CMakeCache.txt" ]; then
#   rm -f CMakeCache.txt
# fi

cmake ../../

make -j32 2>&1 | tee build.log

cd ../../
