# model_finetune_process.py
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
import torch
import gc

class ClearCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # clear the cached memory and trigger the garbage collection
        torch.cuda.empty_cache()
        gc.collect()

def get_trainer(model, train_dataset, tokenizer):
    training_arguments = TrainingArguments(
        output_dir="aniket-math-small-gpt",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        max_steps=1000,
        num_train_epochs=1,
        push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ClearCacheCallback()] 
    )

    return trainer
