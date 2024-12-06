# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Tuple

import torch
import transformers

import vptq
from vptq.ops.gemm import fast_gemv

model_name = "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft"


def infer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    m = vptq.AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    inputs = tokenizer(
        "Explain: Do Not Go Gentle into That Good Night", return_tensors="pt"
    ).to("cuda")
    out = m.generate(**inputs, max_new_tokens=100, pad_token_id=2)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def load_parameters() -> Tuple[torch.Tensor]:
    m = vptq.AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    quantized_params = m.model.state_dict()

    centriods = quantized_params["layers.0.mlp.up_proj.centroids.weight"]
    indices = quantized_params["layers.0.mlp.up_proj.indices"]
    perm = quantized_params["layers.0.mlp.down_proj.perm"]
    weight_scale = quantized_params["layers.0.mlp.down_proj.weight_scale"]
    weight_bias = quantized_params["layers.0.mlp.down_proj.weight_bias"]

    return centriods, indices, perm, weight_scale, weight_bias


def test_fast_gemv(
    num_codebooks: int = 1,
    num_centroids: int = 65536,
    vector_len: int = 8,
    num_res_centroids: int = 1,
    outlier_num_centroids: int = -1,
    outlier_vector_len: int = -1,
    in_features: int = 28672,
    out_features: int = 8192,
):
    centroids, indices, perm, weight_scale, weight_bias = load_parameters()
    x = torch.randn(
        1, 12, 8192, device="cuda", dtype=torch.bfloat16, requires_grad=False
    )
    output = fast_gemv(
        x,
        indices,
        centroids,
        residual_indices=None,
        residual_centroids_=None,
        outlier_indices=None,
        outlier_centroids_=None,
        perm=perm,
        weight_scale=weight_scale,
        weight_bias=weight_bias,
        bias=None,
        num_codebooks=num_codebooks,
        num_centroids=num_centroids,
        vector_len=vector_len,
        num_res_centroids=num_res_centroids,
        outlier_num_centroids=outlier_num_centroids,
        outlier_vector_len=outlier_vector_len,
        in_features=in_features,
        out_features=out_features
    )
    print(output)

    return output


test_fast_gemv()
