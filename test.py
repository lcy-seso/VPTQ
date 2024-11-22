import vptq
import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained(
    "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft")
m = vptq.AutoModelForCausalLM.from_pretrained(
    "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft", device_map='auto')

inputs = tokenizer("Explain: Do Not Go Gentle into That Good Night",
                   return_tensors="pt").to("cuda")
out = m.generate(**inputs, max_new_tokens=100, pad_token_id=2)
print(tokenizer.decode(out[0], skip_special_tokens=True))
