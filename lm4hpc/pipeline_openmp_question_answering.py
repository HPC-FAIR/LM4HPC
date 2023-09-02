import torch, json, os, openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
    elif model == 'gpt-3.5-turbo':
        msg_test = [
        {"role": "system", "content": "You are an OpenMP export."},
        {"role": "user", "content": "What is OpenMP?"},
    ]
        response_test = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=question,
        temperature=0, max_tokens=256
        )
        response_test['choices'][0]['message']['content']
    elif model == 'HuggingFaceH4/starchat-alpha':
        model_id = "HuggingFaceH4/starchat-alpha"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                    load_in_8bit=True,
                                                    device_map='auto'
                                                    )
        def generate_response(input_prompt):
            system_prompt = "<|system|>\nBelow is a conversation between a human user and a helpful AI coding assistant.<|end|>\n"

            user_prompt = f"<|user|>\n{input_prompt}<|end|>\n"

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
        return generate_response(question)
    else:
        raise ValueError('Unknown model: {}'.format(model))
