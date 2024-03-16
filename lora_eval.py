from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
model_id = "google-t5/t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)
data_test = load_dataset('json', data_files='june_version_7921032488/p2d_prompts_test.jsonl')
data_rev = load_dataset('json', data_files='june_version_7921032488/p2d_reverse_prompts.jsonl')

def preprocess_function(examples):
    return tokenizer(
        examples['prompt'],
        examples['completion'],
        max_length=512,
    )


tokenized_data_test = data_test.map(preprocess_function, batched=True, num_proc=4)
tokenized_data_rev = data_rev.map(preprocess_function, batched=True, num_proc=4)

model_test = AutoModelForCausalLM.from_pretrained("saved/")