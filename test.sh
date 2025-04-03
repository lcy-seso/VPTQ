#!/bin/bash

# python setup.py develop
CUDA_VISIBLE_DEVICES=3 python bench/for_profile.py 2>&1 | tee build.log

# CUDA_VISIBLE_DEVICES=3 python bench/bench_quant_gemv.py  2>&1 | tee H100.md

# CUDA_VISIBLE_DEVICES=3 python3 tests/test_quant_gemv.py 2>&1 | tee build.log
