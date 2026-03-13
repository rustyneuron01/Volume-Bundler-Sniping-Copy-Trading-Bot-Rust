def get_tokenizer_with_min_len(model):
    """
    Returns the tokenizer with the smallest maximum token length.

    Args:
        model: Single pipeline or dict of pipeline stages.

    Returns:
        tuple: (tokenizer, max_token_length)
    """
    # Get the model to check for tokenizers
    pipeline = model["stage1"] if isinstance(model, dict) else model

    # If model has two tokenizers, return the one with smaller max length
    if hasattr(pipeline, "tokenizer_2"):
        len_1 = pipeline.tokenizer.model_max_length
        len_2 = pipeline.tokenizer_2.model_max_length
        return (
            (pipeline.tokenizer_2, len_2)
            if len_2 < len_1
            else (pipeline.tokenizer, len_1)
        )

    return pipeline.tokenizer, pipeline.tokenizer.model_max_length


def truncate_prompt_if_too_long(prompt: str, model):
    """
    Truncates the input string if it exceeds the maximum token length when tokenized.

    Args:
        prompt (str): The text prompt that may need to be truncated.

    Returns:
        str: The original prompt if within the token limit; otherwise, a truncated version of the prompt.
    """
    tokenizer, max_token_len = get_tokenizer_with_min_len(model)
    tokens = tokenizer(prompt, verbose=False)  # Suppress token max exceeded warnings
    if len(tokens["input_ids"]) < max_token_len:
        return prompt

    # Truncate tokens if they exceed the maximum token length, decode the tokens back to a string
    truncated_prompt = tokenizer.decode(
        token_ids=tokens["input_ids"][: max_token_len - 1], skip_special_tokens=True
    )
    tokens = tokenizer(truncated_prompt)
    return truncated_prompt
