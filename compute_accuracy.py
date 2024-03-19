def exact_match_accuracy(completions, targets): 
    tot_len = len(completions)
    matched_nums = 0
    for completion, target in zip(completions, targets):
        completion = completion.lower()
        target = target.lower()
        target_len = len(target)
        matched = 0
        for j in range(len(completion) - len(target)):
            if completion[j:j + len(target)] == target:
                matched = 1
        matched_nums += matched
    
    return float(matched_nums)/float(tot_len)