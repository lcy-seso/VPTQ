#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python bench/for_profile.py 2>&1 | tee build.log

# CUDA_VISIBLE_DEVICES=0 python3 tests/test_quant_gemv.py 2>&1 | tee build.log
