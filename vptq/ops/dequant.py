# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
__all__ = [
    'dequant',
]

from typing import Tuple

import torch
import torch.nn as nn

# isort: off
# we need to import the CUDA kernels after importing torch
try:
    from vptq import cuda_ops
except ImportError:
    pass
