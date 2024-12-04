# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
__all__ = [
    'fast_gemv',
    'fused_gemv',
]

from typing import Tuple

import torch
import torch.nn as nn

# isort: off
# we need to import the CUDA kernels after importing torch
# isort: off
# we need to import the CUDA kernels after importing torch
try:
    from vptq import cuda_ops
except ImportError:
    pass


def fast_gemv(
    x: torch.Tensor,
    indices: torch.Tensor,
    centroids: torch.Tensor,
    res_indices: torch.Tensor,
    res_centroids: torch.Tensor,
    outlier_indices: torch.Tensor,
    outlier_centroids: torch.Tensor,
    perm: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    bias: torch.Tensor,
    vector_len: int,
    in_features: int,
    out_features: int,
) -> Tuple[torch.Tensor]:
    pass


def fused_gemv():
    pass
