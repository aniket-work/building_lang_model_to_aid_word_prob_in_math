
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(model, "Simple-Learner/aniket-math-small-gpt", from_transformers=True)
model = peft_model.merge_and_unload()
model.push_to_hub("Simple-Learner/aniket-math-small-gpt")
