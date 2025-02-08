#!/bin/bash

python3 tests/test_quant_gemv.py 2>&1 | tee build.log
