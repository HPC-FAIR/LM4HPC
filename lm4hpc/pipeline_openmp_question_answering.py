import torch
import json
import os
import openai
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from ._instruct_pipeline import InstructionTextGenerationPipeline
# import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
# from htmlTemplates import bot_template, user_template, css

from transformers import pipeline

# Load the configuration once when the module is imported
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)


import os


def get_pdf_text(input_path):
    """
    Extract text from a given PDF file or from all PDF files within a specified directory.

    Parameters:
    - input_path (str): Path to the PDF file or directory containing PDF files.

    Returns:
    - str: Extracted text from the PDF file(s).

    Raises:
    - ValueError: If the provided path is neither a PDF file nor a directory containing PDF files.
    - FileNotFoundError: If the provided path does not exist.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"The specified path '{input_path}' does not exist.")

    # Extract text from a single PDF
    def extract_from_pdf(pdf_path):
        try:
            with PdfReader(pdf_path) as reader:
                return ''.join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"Error reading '{pdf_path}': {e}")
            return ""

    # If the input is a single PDF file
    if os.path.isfile(input_path) and input_path.endswith('.pdf'):
        return extract_from_pdf(input_path)

    # If the input is a directory
    elif os.path.isdir(input_path):
        pdf_files = [os.path.join(input_path, file) for file in os.listdir(
            input_path) if file.endswith('.pdf')]

        if not pdf_files:
            raise ValueError("No PDF files found in the specified directory.")

        return ''.join(extract_from_pdf(pdf_file) for pdf_file in pdf_files)

    else:
        raise ValueError(
            "The provided path is neither a PDF file nor a directory containing PDF files.")


def llm_generate_dolly(model: str, question: str, **parameters) -> str:
    tokenizer_pretrained = AutoTokenizer.from_pretrained(
        model, padding_side="left")
    model_pretrained = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)
    generate_text = InstructionTextGenerationPipeline(
        model=model_pretrained, tokenizer=tokenizer_pretrained, **parameters)
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
                             eos_token_id=0,
                             pad_token_id=0,
                             max_length=256,
                             early_stopping=True)
    output = tokenizer.decode(outputs[0])
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
