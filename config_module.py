# config_module.py
from peft import LoraConfig, mapping

print(mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["self_attn.q_proj", "self_attn.k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
