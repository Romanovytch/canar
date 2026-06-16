def estimate_tokens_and_cost(answer):
    number_of_token = len(answer) / 4
    return number_of_token, round(number_of_token * 0.002, 2)