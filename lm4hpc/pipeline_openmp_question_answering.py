import torch, json, os, openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
        config = json.load(f)

def llm_generate_dolly(model: str, question: str, **parameters) -> str:
    generate_text = pipeline(model = model, **parameters)
    return generate_text(question)[0]["generated_text"].split("\n")[-1]

def llm_generate_gpt(model: str, question: str, **parameters) -> str:
    msg = [{"role": "system", "content": "You are an OpenMP export."}]
    msg.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        model=model,
        messages=msg,
        **parameters
        )
    return response['choices'][0]['message']['content']

def llm_generate_starchat(model: str, question: str, **parameters) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model,
                                                load_in_8bit=True,
                                                device_map='auto'
                                                )
    system_prompt = "<|system|>\nBelow is a conversation between a human user and an OpenMP expert.<|end|>\n"
    user_prompt = f"<|user|>\n{question}<|end|>\n"
    assistant_prompt = "<|assistant|>"
    full_prompt = system_prompt + user_prompt + assistant_prompt
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs,
                                    eos_token_id = 0,
                                    pad_token_id = 0,
                                    max_length=256,
                                    early_stopping=True)
    output =  tokenizer.decode(outputs[0])
    output = output[len(full_prompt):]
    if "<|end|>" in output:
        cutoff = output.find("<|end|>")
        output = output[:cutoff]
    return output
    


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
    if model in config['openmp_question_answering']['models'] and model.startswith('databricks/dolly-v2'):
        response = llm_generate_dolly(model, question, **parameters)
        return response
    elif model == 'gpt-3.5-turbo':
        response = llm_generate_gpt(model, question, **parameters)
        return response
    elif model == 'HuggingFaceH4/starchat-alpha':
        response = llm_generate_starchat(model, question, **parameters)
        return response
    else:
        raise ValueError('Unknown model: {}'.format(model))
