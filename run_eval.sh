sbatch --time=30 -G 1 --cpus-per-task=1 --mem-per-cpu=16g --gres=gpumem:8g --wrap="python /cluster/project/sachan/minjing/peft_knowledge/lora_reverse_eval.py"
