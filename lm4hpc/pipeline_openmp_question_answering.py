import torch, json, os
from transformers import pipeline

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
        config = json.load(f)

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
        generate_text = pipeline(model="databricks/dolly-v2-12b",
                         torch_dtype=torch.bfloat16,
                         trust_remote_code=True,
                         device_map="auto",
                         max_new_tokens=256,
                         temperature=0.001,
                         return_full_text=True)
        return generate_text(question)[0]["generated_text"].split("\n")[-1]
    elif model == 'gpt':
        gptmodel = config["openmp_question_answering"]["default_parameters"]["gpt"]["gptmodel"]
        # initialize gpt3 model
        pass
    elif model == 'starcoder':
        # initialize starcoder model
        pass
    else:
        raise ValueError('Unknown model: {}'.format(model))
