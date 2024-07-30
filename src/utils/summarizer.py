from langchain_community.document_loaders import PyPDFLoader
from src.utils.load_openai import get_response
from src.utils.load_config import LoadConfig
from typing import List, Dict
import requests

CONFIG = LoadConfig()


def read_the_doc(file_dir: List) -> str:
    """This function turns provided files from pdf format to one string."""
    docs = ""
    if isinstance(file_dir, list):
        print("Loading the uploaded documents...")
        document = PyPDFLoader(file_dir[0]).load()
        for page in document:
            docs += page.page_content

    return docs


def doc_preprocess(docs: str) -> List[str]:
    """
    This function preprocess the single text from pdf to pages that are 1900 characters long.
    :param docs: text to be preprocessed
    :return: List of the chunked strings
    """

    # Chunk size chosen based on a model context length and CUDA problems if it more than or equal to 2000
    chunk_size = 1900

    chunks = []
    start = 0
    text_length = len(docs)

    while start < text_length:
        # Calculate the end index
        end = min(start + chunk_size, text_length)

        # Include the remaining part if it's less than chunk_size
        if text_length - end < chunk_size:
            end = text_length

        # Add the chunk to the list
        chunks.append(docs[start:end])

        # Move to the next chunk
        start += chunk_size

    return chunks


def create_prompt(docs: List[str], i: int) -> Dict[str, str]:
    """
    This function uses Sentence Retrieval method
    :param docs: List of chunked strings
    :param i: Chunk index
    :return: Dict to be able to turn it into json format that model expects
    """
    if i == 0:
        prompt = docs[i] + docs[i + 1][:CONFIG.character_overlap]
    # For pages except the first and the last one
    elif i < len(docs) - 1:
        prompt = docs[i - 1][-CONFIG.character_overlap:] + docs[i] + docs[i + 1][:CONFIG.character_overlap]
    else:  # For the last page
        prompt = docs[i - 1][-CONFIG.character_overlap:] + docs[i]

    prompt = {"prompt": prompt}

    return prompt


def summarize_with_open_source_llm(file_dir: List) -> str:
    """
    This function summarizes the content of a PDF file using Open-Source LLM engine.

    Args:
        file_dir (str): path to the PDF file.

    Returns:
        str: The final summarized content.
    """

    flask_endpoint = f"{CONFIG.flask_app_endpoint}/summarize_the_pdf"

    loaded_doc = read_the_doc(file_dir)
    docs = doc_preprocess(loaded_doc)

    full_summary = ""

    # If the document has more than one pages
    if len(docs) > 1:
        for i in range(len(docs)):
            prompt = create_prompt(docs, i)

            response = requests.post(flask_endpoint, json=prompt)
            response_json = response.json()['response']

            full_summary += response_json

    else:  # If the document has only one page
        full_summary = docs[0]

    full_summary = {"prompt": full_summary}

    response = requests.post(flask_endpoint, json=full_summary)
    final_summary = response.json()['response']

    return final_summary


def summarize_openai_gpt(file_dir: List) -> str:
    """
    This function summarizes the content of a PDF file using OpenAI ChatGPT.

    Args:
        file_dir (str): path to the PDF file.

    Returns:
        str: The final summarized content.
    """

    loaded_doc = read_the_doc(file_dir)
    docs = doc_preprocess(loaded_doc)

    full_summary = ""

    # If the document has more than one pages
    if len(docs) > 1:
        max_final_token = int(CONFIG.max_final_token/len(docs)) - CONFIG.token_threshold
        for i in range(len(docs)):
            prompt = create_prompt(docs, i)

            system_role = CONFIG.summarizer_llm_system_role.format(max_final_token)
            user_prompt = (
                "### Text to summarize ###\n"
                f"{prompt}\n\n"
                "### Response ###\n"
            )
            full_summary += get_response(system_role, user_prompt)

    else:  # If the document has only one page
        full_summary = docs[0]

    user_prompt = (
        "### Text to summarize ###\n"
        f"{full_summary}\n\n"
        "### Response ###\n"
    )
    final_summary = get_response(CONFIG.final_summarizer_llm_system_role, user_prompt)
    return final_summary
