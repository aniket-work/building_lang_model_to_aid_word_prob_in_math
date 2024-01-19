
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from model_finetune_process import get_trainer, ClearCacheCallback
from model_loading_module import model
from data_processing_module import tokenized_data
from tokenizer_module import tokenizer

trainer = get_trainer(model, tokenized_data, tokenizer)
trainer.train()
trainer.push_to_hub()
