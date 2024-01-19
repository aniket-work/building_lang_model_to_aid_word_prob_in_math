# model_loading_module.py
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
from peft import get_peft_model
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map={"": 0},
    trust_remote_code=True,
    quantization_config=bnb_config
)

def load_and_configure_model(model, config):
    model = get_peft_model(model, config)
    return model
