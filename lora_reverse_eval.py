from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from datasets import load_dataset
from evaluate import load
model_id = "mins"
model_ids = ["google/flan-t5-base", "google-t5/t5-base", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b", "mistralai/Mistral-7B-Instruct-v0.2"]
tokenizer = AutoTokenizer.from_pretrained(model_ids[2])
data_test = load_dataset('json', data_files='june_version_7921032488/p2d_prompts_test.jsonl')
data_rev = load_dataset('json', data_files='june_version_7921032488/p2d_reverse_prompts_test.jsonl')
tokenizer.pad_token = tokenizer.eos_token
def preprocess_function(examples):
    return tokenizer(
        examples['prompt'],
        # examples['completion'],
        return_tensors="pt",
        max_length=384,
        padding='max_length'
    )


tokenized_data_test = data_test['train'].map(preprocess_function, batched=True)
tokenized_data_rev = data_rev['train'].map(preprocess_function, batched=True)
model_test = AutoModelForCausalLM.from_pretrained(model_ids[2], load_in_4bit=True)
model_test = PeftModel.from_pretrained(model_test, 'saved')

outputs_test = model_test.generate(tokenized_data_test, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
outputs_rev = model_test.generate(tokenized_data_rev, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

decoded_test = tokenizer.batch_decode(outputs_test, skip_special_tokens=True)
decoded_rev = tokenizer.batch_decode(outputs_rev, skip_special_tokens=True)
exact_match = load("exact_match")
res_test = exact_match.compute(decoded_test, data_test['completion'])
res_rev = exact_match.compute(decoded_rev, data_rev['completion'])

print(res_test, res_rev)

