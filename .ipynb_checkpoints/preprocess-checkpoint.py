from transformers import AutoTokenizer, T5Tokenizer

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


tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    anss = [', '.join(a) for a in examples['candidates']]
    questions = [q + " Choose one of the following as an answer: " + a for (q,a) in zip(questions, anss)]
    contexts = [' '.join(s) for s in examples['supports']]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs