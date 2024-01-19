# data_processing_module.py
from datasets import load_dataset, Dataset
from tokenizer_module import tokenizer

def tokenize(sample):
    model_inps = tokenizer(sample["text"], padding=True, truncation=True, max_length=256)
    return model_inps

data = load_dataset("gsm8k", "main", split="train")
data_df = data.to_pandas()
data_df["text"] = data_df[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)
data = Dataset.from_pandas(data_df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)
