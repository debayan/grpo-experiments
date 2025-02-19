This repository is for experiments using the GRPO RL algorithm, trying to train a small LLM to produce valid SPARQL queries on DBLP Knowledge Graph.

How to run?

git clone this repository.

python3 -m venv .

source bin/activate

pip install -r requirements.txt

python3 -u train_sparql_grpo.py
