from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.utils.load_openai import get_response
from src.utils.load_config import LoadConfig
from langchain_chroma import Chroma
from typing import List, Tuple
import requests
import time
import os

CONFIG = LoadConfig()


def get_retrieved_content(chatbot: List, message: str, data_type: str):
    """This function uses a vector database to retrieve content that is similar to the user prompt."""
    embedding_model = SentenceTransformerEmbeddings(model_name=CONFIG.embedding_model_config)

    if data_type == "Preprocessed doc":
        # directories
        if os.path.exists(CONFIG.persist_directory):
            vectordb = Chroma(persist_directory=CONFIG.persist_directory,
                              embedding_function=embedding_model)
        else:
            chatbot.append(
                (message, f"VectorDB does not exist.")
            )
            return "", chatbot, None

    elif data_type == "Upload doc: Process for RAG":
        if os.path.exists(CONFIG.custom_persist_directory):
            vectordb = Chroma(persist_directory=CONFIG.custom_persist_directory,
                              embedding_function=embedding_model)
        else:
            chatbot.append(
                (message, f"No file was uploaded. Please first upload your files using the 'upload' button.")
            )
            return "", chatbot, None

    docs = vectordb.similarity_search(message, k=3)
    retrieved_docs_page_content = [str(x.page_content) + "\n\n" for x in docs]
    retrieved_content = "# Retrieved content:\n\n" + str(retrieved_docs_page_content)

    return "", chatbot, retrieved_content


def respond(chatbot: List, message: str, model_name: str, data_type: str = "Preprocessed doc") -> Tuple:
    """
    Generate a response to a user query using document retrieval and language model completion.

    Parameters:
        chatbot (List): List representing the chatbot's conversation history.
        message (str): The user's query.
        model_name (str): The model name.
        data_type (str): Type of data used for document retrieval ("Preprocessed doc" or "Upload doc: Process for RAG").

    Returns:
        Tuple: A tuple containing an empty string, the updated chat history, and references from retrieved documents.
    """
    flask_endpoint = f"{CONFIG.flask_app_endpoint}/generate_response"

    input_txt, chatbot, retrieved_content = get_retrieved_content(chatbot, message, data_type)

    chat_history = f"Chat history: \n {str(chatbot[-2:])}\\n\n"

    if model_name == "Open-Source model Mistral7b":
        prompt_template = (CONFIG.llm_template.format(chat_history, retrieved_content, message))

        data = {
            "prompt": prompt_template,
            "max_new_tokens": CONFIG.max_new_tokens,
            "temperature": CONFIG.temperature,
        }

        response = requests.post(flask_endpoint, json=data)
        response = response.json()['response']

    else:
        user_message = CONFIG.user_message.format(chat_history, retrieved_content, message)
        response = get_response(CONFIG.system_message, user_message)

    chatbot.append(
        (message, response)
    )
    time.sleep(2)

    return "", chatbot, retrieved_content
