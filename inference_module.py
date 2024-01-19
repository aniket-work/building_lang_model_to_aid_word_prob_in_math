# inference_module.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Simple-Learner/aniket-math-small-gpt", trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
inputs = tokenizer('''question: I have 2 apples. My friend gave me another two apples. I ate 1 apple. Totally how many I have now? answer: ''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=512)
text = tokenizer.batch_decode(outputs)[0]
print(text)
