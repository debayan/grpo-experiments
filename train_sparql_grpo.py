# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import sys
from difflib import SequenceMatcher
import wandb

dataset = load_dataset("json", data_files="DBLP-QuAD/dblpquad_sparql_train_1.json")["train"]
print(dataset[0])

def reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for c,g in zip(completions,ground_truth):
        c = c.replace(' ','').lower()
        g = g.replace(' ','').lower()
        rewards.append(SequenceMatcher(None, c, g).ratio())
    return rewards

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10, use_vllm=True, vllm_gpu_memory_utilization=0.5, report_to="wandb")
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.model.to("cuda")
trainer.train()
