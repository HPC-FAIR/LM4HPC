
import torch
from transformers import pipeline

def openmp_question_answering(model, question):


    text_generator = pipeline(model="databricks/dolly-v2-12b",
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True,
                            device_map="auto",
                            max_new_tokens=256,
                            temperature=0.001,
                            return_full_text=True)
    return text_generator
def openmp_question_answering(model, question):
    if model == 'databricks/dolly-v2-12b':
        text_generator = pipeline(model=model,
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True,
                                device_map="auto",
                                max_new_tokens=256,
                                temperature=0.001,
                                return_full_text=True)
        return text_generator(question)[0]["generated_text"].split("\n")[-1]
    elif model == 'gpt3':
        # initialize gpt3 model
        pass
    elif model == 'starcoder':
        # initialize starcoder model
        pass
    else:
        raise ValueError('Unknown model: {}'.format(model))
