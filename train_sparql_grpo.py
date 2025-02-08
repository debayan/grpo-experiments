# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import sys
from difflib import SequenceMatcher
import wandb

dataset = load_dataset("json", data_files="DBLP-QuAD/dblpquad_sparql_train_1.json")["train"]
print(dataset[0])

from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.sparql import Query

def is_valid_sparql(query):
    """Check if a SPARQL query is syntactically valid using rdflib."""
    try:
        parsed_query = parseQuery(query)
        if isinstance(parsed_query, Query):
            print("valid sparql:",query)
            return 1
        else:
            return -1
    except Exception as e:
        print(f"Invalid SPARQL query: {e}")
    return -1

def reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for c, g in zip(completions, ground_truth):
        extracted_c = c.replace(' ', '').lower()
        g = g.replace(' ', '').lower()
        similarity_reward = SequenceMatcher(None, extracted_c, g).ratio()
        length_reward = -abs(float(len(c)-len(g))/len(g))
        valid_sparql_reward = is_valid_sparql(c) 
        rewards.append(similarity_reward+length_reward+valid_sparql_reward)
    print("rewards:",rewards)
    for c, g in zip(completions[:1], ground_truth[:1]):
        print("Printing 1 samples")
        print("completion:",c)
        print("ground_truth:", g)
        print("===========================")
    return rewards

training_args = GRPOConfig(output_dir="Qwen2-1.5B-GRPO-continue", logging_steps=10,  report_to="wandb", max_completion_length=128, per_device_eval_batch_size = 4, gradient_accumulation_steps=8, num_generations=4, temperature=0.1, num_train_epochs=10, use_vllm=True, vllm_gpu_memory_utilization=0.5)
trainer = GRPOTrainer(
#    model="Qwen/Qwen2-1.5B-Instruct",
    model="./Qwen2-1.5B-GRPO/checkpoint-5250/",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.model.to("cuda")
trainer.train()
