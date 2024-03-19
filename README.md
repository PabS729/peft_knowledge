# Peft Knowledge Project

## Overview

v1.0
LoRA on Mistral-7B for Reversal Curse dataset

### Setup

pip install -r requirements.txt

### Execution

Running Training Script on Euler:

./run_train.sh

Inference on Euler:

./run_eval.sh

## Updates
03/19/2024: Rewritten data preprocessing and training to match that from Experiment 1 of the Reversal Curse Paper, evaluation in progress. 

03/13/2024: Current result for evaluating PEFT LLM is not good. Needs further finetuning or different model choices.
