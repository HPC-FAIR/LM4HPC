import torch
from transformers import pipeline

def openmp_question_answering(model: str, question: str, **parameters) -> str:
    """
    Generates an answer to a question using the specified model and parameters.

    Parameters:
        model (str): The model to use for question answering. Options are 'databricks/dolly-v2-12b', 'gpt3', and 'starcoder'.
        question (str): The question to answer.
        **parameters: Additional keyword arguments to pass to the `pipeline` function.

    Returns:
        str: The generated answer.

    Raises:
        ValueError: If the model is not valid.
    """
    if model == 'databricks/dolly-v2-12b':
        text_generator = pipeline(model=model, **parameters)
        return text_generator(question)[0]["generated_text"].split("\n")[-1]
    elif model == 'gpt3':
        # initialize gpt3 model
        pass
    elif model == 'starcoder':
        # initialize starcoder model
        pass
    else:
        raise ValueError('Unknown model: {}'.format(model))
