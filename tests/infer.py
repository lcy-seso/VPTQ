# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Tuple

import torch
import transformers

import pdb

import vptq

model_name = "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft"


def infer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    m = vptq.AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto')

    inputs = tokenizer(
        "Explain: Do Not Go Gentle into That Good Night", return_tensors="pt"
    ).to("cuda")
    out = m.generate(**inputs, max_new_tokens=100, pad_token_id=2)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


infer()
