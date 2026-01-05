import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen3-4B"

print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",                # 关键：强制全上 GPU（4bit 才现实）
    quantization_config=bnb_cfg,
)

print("loaded 4bit ok")
