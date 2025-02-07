import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model
output_dir = "Qwen2-0.5B-GRPO/checkpoint-1311"  # Directory where model was saved
model_name = "Qwen/Qwen2-0.5B-Instruct"  # Base model name

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir)
model.to("cuda")  # Move to GPU

# Load dataset for inference (optional, modify as needed)
dataset = load_dataset("json", data_files="DBLP-QuAD/dblpquad_sparql_train_1.json")["train"]

# Define test inputs (modify as needed)
print(dataset[0]["prompt"])
test_inputs = [example["prompt"] for example in dataset.select(range(10))]
ground_truths = [example["ground_truth"] for example in dataset.select(range(10))]

# Generate outputs
outputs = []
for inp in test_inputs:
    inputs = tokenizer(inp, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=1024)  # Adjust max_length as needed
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    outputs.append(output_text)

# Compute rewards using the reward function
from difflib import SequenceMatcher

def reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for c, g in zip(completions, ground_truth):
        c = c.replace(" ", "").lower()
        g = g.replace(" ", "").lower()
        rewards.append(SequenceMatcher(None, c, g).ratio())
    return rewards

rewards = reward_func(outputs, ground_truths)

# Save outputs to a JSON file
output_data = [{"input": inp, "output": out, "ground_truth": gt, "reward": r} 
               for inp, out, gt, r in zip(test_inputs, outputs, ground_truths, rewards)]

with open("model_outputs.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("Inference completed. Outputs saved to model_outputs.json")

