from transformers import AutoTokenizer, T5Tokenizer

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
def preprocess_function(examples):
    questions = examples["question"]
    # print(len(questions))
    contexts = [' '.join(s) for s in examples['supports']]
    anss = [', '.join(c) for c in examples['candidates']]
    questions = [q + " Choose one of the following as an answer: " + a for (q,a) in zip(questions, anss)]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        return_offsets_mapping=True,
        # padding="max_length"
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    length_mapping = [i[-2][1] for i in offset_mapping]
    start_positions = []
    end_positions = []
    ans_starts = []

    pos = 0
    found = False
    for c in range(len(contexts)):
        length = len(answers[c])
        for r in range(len(contexts[c])):
            # print(contexts[c])
            if contexts[c][r].lower() == answers[c][0].lower() and r < len(contexts[c]) - length:
                next_str = contexts[c][r:r+length].lower()
                if next_str == answers[c]:
                    pos += r
                    found = True
                    break
        if found == False:
            ans_starts.append(0)
        else:
            ans_starts.append(pos)
            found = False
        pos = 0
    # print(ans_starts[:100])
    new_k = []
    for a,s in zip(answers, ans_starts):
        new_k.append({"text": [a], 'answer_start': [s]})
    # input['answers'] = new_k
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = ans_starts[i]
        end_char = ans_starts[i] + len(answer)
        
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    # print(len(offset_mapping))
    # print(start_positions[:50])

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = start_positions
    # print(len(contexts), len(questions), len(answers), inputs[])
    return inputs

def preprocess_validation(examples):
    questions = examples['question']
    contexts = [' '.join(s) for s in examples['supports']]
    anss = [', '.join(c) for c in examples['candidates']]
    questions = [q + " Choose one of the following as an answer: " + a for (q,a) in zip(questions, anss)]
    answers = examples['answer']

    
    examples['context'] = contexts

    ans_starts = []

    pos = 0
    found = False
    for c in range(len(contexts)):
        length = len(answers[c])
        for r in range(len(contexts[c])):
            # print(contexts[c])
            if contexts[c][r].lower() == answers[c][0].lower() and r < len(contexts[c]) - length:
                next_str = contexts[c][r:r+length].lower()
                if next_str == answers[c]:
                    pos += r
                    found = True
                    break
        if found == False:
            ans_starts.append(0)
        else:
            ans_starts.append(pos)
            found = False
        pos = 0
    examples['question'] = questions
    examples['answers'] = [{"ans_start": [a], "text": [b]} for a, b in zip(ans_starts, answers)]
    return examples