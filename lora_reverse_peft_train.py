import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForQuestionAnswering, T5Tokenizer, pipeline
from wikihop import WikiHop
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
# from preprocess import * 
from postprocess import * 
from datasets import load_metric, load_dataset
import tqdm

model_id_flan = "google/flan-t5-base"
model_ids = ["google/flan-t5-base", "google-t5/t5-base", "Mistral-7B-v0.1", "meta-llama/Llama-2-7b", "mistralai/Mistral-7B-Instruct-v0.2"]
model_id_n = "google-t5/t5-base"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_ids[2], quantization_config=bnb_config, device_map={"":0})
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'], 
    lora_dropout=0.05, 
    bias="none", 
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



data_train = load_dataset('json', data_files='june_version_7921032488/p2d_prompts_train.jsonl')
print(data_train.keys())

tokenizer = AutoTokenizer.from_pretrained(model_ids[2])
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(
        text=examples['prompt'],
        text_target=examples['completion'],
        max_length=384,
        padding="max_length",
    )

def main():
    data_train.cleanup_cache_files()
    tokenized_data_train = data_train['train'].map(preprocess_function, batched=True, remove_columns=data_train['train'].column_names)
    # tokenized_data = data.map(preprocess_function, batched=True, remove_columns=data["train"].column_names)
    # tokenized_data_dev = data_dev.map(preprocess_function, batched=True, remove_columns=data_train["train"].column_names)
    print(len(tokenized_data_train))
    # data = dt.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    import transformers

    # needed for gpt-neo-x tokenizer
    # tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_data_train,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=5,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            # remove_unused_columns=False
        ),
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    trainer.save_model("saved_eu")


if __name__ == "__main__":
    main()
    # data.cleanup_cache_files()
    # data_val = data["validation"].map(preprocess_validation, batched=True)

    # print(data_val[0])

    # model_test = AutoModelForCausalLM.from_pretrained()