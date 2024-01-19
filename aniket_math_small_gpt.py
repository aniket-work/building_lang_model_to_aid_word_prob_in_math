# aniket_math_small_gpt.py

from tokenizer_module import tokenizer
from bits_and_bytes_config_module import bnb_config
from model_loading_module import model, load_and_configure_model, get_peft_model
from config_module import config
from data_processing_module import tokenized_data
from training_module import trainer
from peft_model_module import model as peft_model
from inference_module import model as inference_model, tokenizer as inference_tokenizer

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, mapping
from peft_model_module import PeftModel
from model_finetune_process import get_trainer

import torch
from transformers import TrainerCallback
import gc


for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

print(mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING)

model = get_peft_model(model, config)

def tokenize(sample):
    model_inps = tokenizer(sample["text"], padding=True, truncation=True, max_length=256) 
    return model_inps

data = load_dataset("gsm8k", "main", split="train")
data_df = data.to_pandas()
data_df["text"] = data_df[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)
data = Dataset.from_pandas(data_df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)

# Use the trainer function from the training module
trainer = get_trainer(model, tokenized_data, tokenizer)
trainer.train()
trainer.push_to_hub()

# Use PeftModel class from the peft_model_module
peft_model = PeftModel.from_pretrained(model, "Simple-Learner/aniket-math-small-gpt", from_transformers=True)
model = peft_model.merge_and_unload()
model.push_to_hub()


