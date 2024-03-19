from transformers import AutoTokenizer
import copy
model_ids = ["google/flan-t5-base", "google-t5/t5-base", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b", "mistralai/Mistral-7B-Instruct-v0.2"]
tokenizer = AutoTokenizer.from_pretrained(model_ids[2])
tokenizer.pad_token = tokenizer.eos_token


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def preprocess_function_dec(examples, tokenizer, predict_with_generate=False):
    if predict_with_generate:
        inputs = [doc for doc in examples["prompt"]]
    else:
        inputs = [doc + ex for doc, ex in zip(examples["prompt"], examples["completion"])]

    model_inputs = tokenizer(inputs)
    assert "attention_mask" in model_inputs

    # TODO: think how to add labels and compute the loss even with `predict_with_generate`
    model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])

    # if wandb.config.ignore_loss_on_prompt_tokens:
    #     prompts = [tokenizer.encode(doc) for doc in examples["prompt"]]
    #     prompt_lengths = [len(prompt) for prompt in prompts]
    #     for j, label in enumerate(model_inputs["labels"]):
    #         for i in range(0, prompt_lengths[j]):
    #             label[i] = -100

    return model_inputs


def max_pad_evaluate(
    examples,
    tokenizer,
    max_pad_length,
    keys_to_pad=["input_ids", "attention_mask", "labels"],
):
    # Due to the way that tensors are concatenated during evaluation, we need to pad the inputs to the max length of the batch

    for key in keys_to_pad:
        examples_key_batch = [e for e in examples[key]]
        padding_value = None
        if key == "labels":
            padding_value = -100
        elif key == "attention_mask":
            padding_value = 0
        else:
            padding_value = tokenizer.pad_token_id
        examples_key_batch_padded = [[padding_value] * (max_pad_length - len(e)) + e for e in examples_key_batch]
        examples[key] = examples_key_batch_padded

    return examples

def preprocess_training(examples):
    return preprocess_function_dec(examples, tokenizer=tokenizer)

def preprocess_with_generate(examples):
    return preprocess_function_dec(examples, tokenizer=tokenizer, predict_with_generate=True)

def max_pad_function_curried(max_length):
    return lambda examples: max_pad_evaluate(examples, tokenizer, max_length)