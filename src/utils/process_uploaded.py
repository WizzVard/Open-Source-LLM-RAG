from src.utils.summarizer import summarize_with_open_source_llm, summarize_openai_gpt
from src.utils.prepare_vectordb import PrepareVectorDB
from src.utils.load_config import LoadConfig
from typing import List, Tuple
import time

CONFIG = LoadConfig()


def process_uploaded_files(files_dir: List, chatbot: List, rag_with_dropdown: str, model_name: str) -> Tuple:
    """
    Process uploaded files to prepare a VectorDB

    Parameters:
        files_dir (List): List of paths to the uploaded files.
        chatbot: An instance of the chatbot for communication.

    Returns:
        Tuple: A tuple containing an empty string and the updated chatbot instance.
    """

    if rag_with_dropdown == "Upload doc: Process for RAG":
        prepare_vectordb_instance = PrepareVectorDB(data_directory=files_dir,
                                                    persist_directory=CONFIG.custom_persist_directory,
                                                    embedding_model_config=CONFIG.embedding_model_config,
                                                    chunk_size=CONFIG.chunk_size,
                                                    chunk_overlap=CONFIG.chunk_overlap)
        prepare_vectordb_instance.prepare_and_save_vectordb()
        chatbot.append((" ", "Uploaded files are ready. Please ask your question"))
    elif rag_with_dropdown == "Upload doc: Summary" and model_name == "Open-Source model Mistral7b":
        time.sleep(1)
        final_summary = summarize_with_open_source_llm(file_dir=files_dir)
        chatbot.append(("", final_summary))
    elif rag_with_dropdown == "Upload doc: Summary" and model_name == "gpt-4o-mini":
        time.sleep(1)
        final_summary = summarize_openai_gpt(file_dir=files_dir)
        chatbot.append(("", final_summary))
    else:
        chatbot.append((" ", "If you would like to upload a PDF, "
                             "please select your desired action in 'rag_with' dropdown."))

    return "", chatbot
