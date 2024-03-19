from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, BitsAndBytesConfig
from peft import PeftModel
from compute_accuracy import exact_match_accuracy
from datasets import load_dataset
from evaluate import load
import pandas as pd
import torch
import json
def load_from_jsonl(file_name: str):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data

max_steps = 50
max_length = 300
model_id = "mins"
model_ids = ["google/flan-t5-base", "google-t5/t5-base", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b", "mistralai/Mistral-7B-Instruct-v0.2"]
tokenizer = AutoTokenizer.from_pretrained(model_ids[2])

data_test_all = load_dataset('json', data_files='june_version_7921032488/p2d_prompts_test.jsonl')['train']
data_rev_all = load_dataset('json', data_files='june_version_7921032488/p2d_reverse_prompts_test.jsonl')['train']
data_test = data_test_all['prompt'][:max_length]
data_test_ans = data_test_all['completion'][:max_length]
data_rev = data_rev_all['prompt'][:max_length]
data_rev_ans = data_rev_all['completion'][:max_length]
tokenizer.pad_token = tokenizer.eos_token
print(data_test, data_test_ans, data_rev, data_rev_ans)
# def preprocess_function(examples):
#     return tokenizer(
#         examples,
#         # examples['completion'],
#         return_tensors="pt",
#         # max_length=2048,
#         padding='longest'
#     )
tokenized_data_test = [tokenizer(
        i,
        # examples['completion'],
        return_tensors="pt",
        # max_length=2048,
        # padding='longest'
    ).input_ids.to('cuda') for i in data_test]
tokenized_data_rev = [tokenizer(
        i,
        # examples['completion'],
        return_tensors="pt",
        # max_length=2048,
        # padding='longest'
    ).input_ids.to('cuda') for i in data_rev]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_test = AutoModelForCausalLM.from_pretrained(model_ids[2], quantization_config=bnb_config, device_map={"":0})
model_test = PeftModel.from_pretrained(model_test, "saved_" + model_ids[2]  + "_" + str(max_steps))
outputs_test = []
outputs_rev = []
for tok in tokenized_data_test:
    out = model_test.generate(tok, max_new_tokens=128, num_beams=1, do_sample=False)
    decoded_test = tokenizer.batch_decode(out, skip_special_tokens=True)
    outputs_test.append(decoded_test[0])
# print(outputs_test[:10])
# print(tokenized_data_rev['train'][0])
for tok in tokenized_data_rev:
    out = model_test.generate(tok, max_new_tokens=30, num_beams=1, do_sample=False)
    decoded_rev = tokenizer.batch_decode(out, skip_special_tokens=True)
    outputs_rev.append(decoded_rev[0])

# outputs_rev = model_test.generate(tokenized_data_rev, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

# decoded_rev = tokenizer.batch_decode(outputs_rev, skip_special_tokens=True)
accuracy_test = exact_match_accuracy(outputs_test, data_test_ans)
accuracy_rev = exact_match_accuracy(outputs_rev, data_rev_ans)
print(accuracy_test, accuracy_rev)
# res_test = exact_match.compute(outputs_test, data_test_ans)
# res_rev = exact_match.compute(outputs_rev, data_rev_ans)
dic_test = {"ans": outputs_test}
dic_rev = {'ans': outputs_rev}
df_t = pd.DataFrame.from_dict(dic_test)
df_r = pd.DataFrame.from_dict(dic_rev)

df_t.to_csv("test_answers/test" + "_" + str(max_steps) + ".csv")
df_r.to_csv("test_answers/rev" + "_" + str(max_steps) + ".csv")
# print(res_test, res_rev)

