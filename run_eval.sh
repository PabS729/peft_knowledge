module load gcc/8.2.0 cuda/11.6.2 python_gpu/3.11.2
source peft_knowledge/bin/activate
sbatch --time=30 -G 1 --cpus-per-task=1 --mem-per-cpu=16g --gres=gpumem:8g --wrap="python /cluster/project/sachan/minjing/peft_knowledge/lora_reverse_eval.py"
