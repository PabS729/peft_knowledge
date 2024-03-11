import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForQuestionAnswering, T5Tokenizer
from wikihop import WikiHop
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from preprocess import * 
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
print(data['train'][0])
tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)

# data = dt.map(lambda samples: tokenizer(samples["quote"]), batched=True)

import transformers

# needed for gpt-neo-x tokenizer
# tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()