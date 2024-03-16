import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForQuestionAnswering, T5Tokenizer, pipeline
from wikihop import WikiHop
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from preprocess import * 
from postprocess import * 
from datasets import load_metric
import tqdm

model_id = "EleutherAI/gpt-neox-20b"
model_id_n = "google-t5/t5-base"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = AutoModelForQuestionAnswering.from_pretrained(model_id_n, quantization_config=bnb_config, device_map={"":0})
# name = model.load_adaptor("AdapterHub/bert-base-uncased-pf-wikihop", source="hf")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=['q', 'v'], 
    lora_dropout=0.05, 
    bias="none", 
    task_type=TaskType.QUESTION_ANS
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

from datasets import load_dataset

data = load_dataset('wikihop.py')
print(data.keys())
# data.cleanup_cache_files()
tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)
# tokenized_data_dev = data_dev.map(preprocess_function, batched=True, remove_columns=data_train["train"].column_names)
print(len(tokenized_data['train']))
# data = dt.map(lambda samples: tokenizer(samples["quote"]), batched=True)

import transformers

# needed for gpt-neo-x tokenizer
# tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False
    ),
    data_collator=transformers.DefaultDataCollator(),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
# trainer.train()

trainer.save_model("aa")


import numpy as np
# data.cleanup_cache_files()
data_val = data["validation"].map(preprocess_validation, batched=True)

print(data_val[0])
# validation_features = data_val.map(
#     prepare_validation_features,
#     batched=True,
# )

# raw_predictions = trainer.predict(validation_features)
# validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
# max_answer_length = 30

# offset_mapping = validation_features[0]["offset_mapping"]
# The first feature comes from the first example. For the more general case, we will need to be match the example_id to
# an example index
pips = pipeline("question-answering", model="aa/", device = 'cuda')
for c in tqdm(pips(question=data_val['question'], context=data_val['context'], question_first=True, max_seq_len=4096,)):
    print(c)
# import collections
# from tqdm.auto import tqdm
# squad_v2 = True
# def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
#     all_start_logits, all_end_logits = raw_predictions
#     # Build a map example to its corresponding features.
#     example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
#     features_per_example = collections.defaultdict(list)
#     for i, feature in enumerate(features):
#         features_per_example[example_id_to_index[feature["example_id"]]].append(i)

#     # The dictionaries we have to fill.
#     predictions = collections.OrderedDict()

#     # Logging.
#     print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

#     # Let's loop over all the examples!
#     for example_index, example in enumerate(tqdm(examples)):
#         # Those are the indices of the features associated to the current example.
#         feature_indices = features_per_example[example_index]

#         min_null_score = None # Only used if squad_v2 is True.
#         valid_answers = []

#         context = example["context"]
#         # Looping through all the features associated to the current example.
#         for feature_index in feature_indices:
#             # We grab the predictions of the model for this feature.
#             start_logits = all_start_logits[feature_index]
#             end_logits = all_end_logits[feature_index]
#             # This is what will allow us to map some the positions in our logits to span of texts in the original
#             # context.
#             offset_mapping = features[feature_index]["offset_mapping"]

#             # Update minimum null prediction.
#             cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
#             feature_null_score = start_logits[cls_index] + end_logits[cls_index]
#             if min_null_score is None or min_null_score < feature_null_score:
#                 min_null_score = feature_null_score

#             # Go through all possibilities for the `n_best_size` greater start and end logits.
#             start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
#             end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
#                     # to part of the input_ids that are not in the context.
#                     if (
#                         start_index >= len(offset_mapping)
#                         or end_index >= len(offset_mapping)
#                         or offset_mapping[start_index] is None
#                         or offset_mapping[end_index] is None
#                     ):
#                         continue
#                     # Don't consider answers with a length that is either < 0 or > max_answer_length.
#                     if end_index < start_index or end_index - start_index + 1 > max_answer_length:
#                         continue

#                     start_char = offset_mapping[start_index][0]
#                     end_char = offset_mapping[end_index][1]
#                     valid_answers.append(
#                         {
#                             "score": start_logits[start_index] + end_logits[end_index],
#                             "text": context[start_char: end_char]
#                         }
#                     )

#         if len(valid_answers) > 0:
#             best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
#         else:
#             # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
#             # failure.
#             best_answer = {"text": "", "score": 0.0}

#         # Let's pick our final answer: the best one or the null answer (only for squad_v2)
#         if not squad_v2:
#             predictions[example["id"]] = best_answer["text"]
#         else:
#             answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
#             predictions[example["id"]] = answer

#     return predictions

# final_predictions = postprocess_qa_predictions(data_val, validation_features, raw_predictions.predictions)
# metric = load_metric("squad_v2" if squad_v2 else "squad")

# if squad_v2:
#     formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
# else:
#     formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
# references = [{"id": ex["id"], "answers": ex["answers"]} for ex in data_val]
# metric.compute(predictions=formatted_predictions, references=references)