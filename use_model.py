import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from prettytable import PrettyTable

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained("Simple-Learner/aniket-math-small-gpt", trust_remote_code=True, torch_dtype=torch.float16)

model.to(device)

# Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Get the question from command line input
question = sys.argv[1] if len(sys.argv) > 1 else "No question provided."

# Move the input tensor to the same device as the model
inputs_text = f'question: {question} answer: '
inputs = tokenizer(inputs_text, return_tensors="pt", return_attention_mask=False)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate outputs on the same device
# Set a higher temperature for more diverse outputs
outputs = model.generate(**inputs, max_length=150, temperature=0.2, do_sample=True, repetition_penalty=1.2)

# Decode and print the text
response_text = tokenizer.batch_decode(outputs)[0]

# Display input and output in a table
table = PrettyTable([""])
table.add_row(["Input Question"])
table.add_row(["--------------------------------------------------------------------"])
table.add_row([inputs_text])
table.add_row(["--------------------------------------------------------------------"])
table.add_row(["Model Response"])
table.add_row(["--------------------------------------------------------------------"])
table.add_row([response_text])
print(table)
