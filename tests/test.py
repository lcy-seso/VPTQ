# import vptq
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = "VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v8-k65536-256-woft"

# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model)
# m = vptq.AutoModelForCausalLM.from_pretrained(
#     model, device_map='auto')

# inputs = tokenizer("Explain: Do Not Go Gentle into That Good Night",
#                    return_tensors="pt").to("cuda")
# out = m.generate(**inputs, max_new_tokens=100, pad_token_id=2)
# print(tokenizer.decode(out[0], skip_special_tokens=True))

import transformers

import vptq

model = "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft"

tokenizer = transformers.AutoTokenizer.from_pretrained(model)
m = vptq.AutoModelForCausalLM.from_pretrained(model, device_map="auto")

prompt = "Explain: Do Not Go Gentle into That Good Night"
out = m.generate(
    **tokenizer(prompt, return_tensors="pt").to("cuda"),
    max_new_tokens=100,
    pad_token_id=2
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
