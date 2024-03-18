from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from datasets import load_dataset
from evaluate import load
import pandas as pd
model_id = "mins"
model_ids = ["google/flan-t5-base", "google-t5/t5-base", "Mistral-7B-v0.1", "meta-llama/Llama-2-7b", "mistralai/Mistral-7B-Instruct-v0.2"]
tokenizer = AutoTokenizer.from_pretrained(model_ids[2])
data_test_all = load_dataset('json', data_files='june_version_7921032488/p2d_prompts_test.jsonl')['train']
data_rev_all = load_dataset('json', data_files='june_version_7921032488/p2d_reverse_prompts_test.jsonl')['train']
data_test = data_test_all['prompt']
data_test_ans = data_test_all['completion']
data_rev = data_rev_all['prompt']
data_rev_ans = data_rev_all['completion']
tokenizer.pad_token = tokenizer.eos_token
def preprocess_function(examples):
    return tokenizer(
        examples,
        # examples['completion'],
        return_tensors="pt",
        max_length=384,
        padding='max_length'
    )


tokenized_data_test = [tokenizer(
        i,
        # examples['completion'],
        return_tensors="pt",
        max_length=384,
        padding='max_length'
    ).input_ids for i in data_test]
tokenized_data_rev = [tokenizer(
        i,
        # examples['completion'],
        return_tensors="pt",
        max_length=384,
        padding='max_length'
    ).input_ids for i in data_rev]
model_test = AutoModelForCausalLM.from_pretrained(model_ids[2], load_in_4bit=True)
model_test = PeftModel.from_pretrained(model_test, 'saved')
outputs_test = []
outputs_rev = []
for tok in tokenized_data_test:
    out = model_test.generate(tok, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    decoded_test = tokenizer.batch_decode(out, skip_special_tokens=True)
    outputs_test.append(decoded_test[0])
# print(tokenized_data_rev['train'][0])
for tok in tokenized_data_rev:
    out = model_test.generate(tok, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    decoded_rev = tokenizer.batch_decode(out, skip_special_tokens=True)
    outputs_rev.append(decoded_rev[0])

# outputs_rev = model_test.generate(tokenized_data_rev, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

# decoded_rev = tokenizer.batch_decode(outputs_rev, skip_special_tokens=True)
exact_match = load("exact_match")
res_test = exact_match.compute(outputs_test, data_test_ans)
res_rev = exact_match.compute(outputs_rev, data_rev_ans)
dic_test = {"ans": outputs_test}
dic_rev = {'ans': outputs_rev}
df_t = pd.DataFrame.from_dict(dic_test)
df_r = pd.DataFrame.from_dict(dic_rev)

df_t.to_csv("test_ans.csv")
df_r.to_csv("rev_ans.csv")
# print(res_test, res_rev)

