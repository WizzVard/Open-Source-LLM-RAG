from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import os


class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB using embeddings.

    This class facilitates the process of loading documents, chunking them, and creating a VectorDB
    with embeddings. It provides methods to prepare and save the VectorDB.

    Parameters:
        data_directory (str or List[str]): The directory or list of directories containing the documents.
        persist_directory (str): The directory to save the VectorDB.
        embedding_model_config (str): The engine for embeddings.
        chunk_size (int): The size of the chunks for document processing.
        chunk_overlap (int): The overlap between chunks.
    """

    def __init__(self, data_directory: List, persist_directory: str, embedding_model_config: str,
                 chunk_size: int, chunk_overlap: int) -> None:
        self.embedding_model_config = embedding_model_config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_config)

    def __load_all_documents(self) -> List[Document]:
        """
        Load all documents from the specified directory or directories.

        :return:
            List[Document]: A list of loaded documents
        """
        doc_counter = 0
        docs = []
        if isinstance(self.data_directory, list):
            print("Loading the uploaded documents...")
            for doc_dir in self.data_directory:
                docs.extend(PyPDFLoader(doc_dir).load())
                doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")
        else:
            print("Loading documents manually...")
            document_list = os.listdir(self.data_directory)
            for doc_name in document_list:
                docs.extend(PyPDFLoader(os.path.join(
                    self.data_directory, doc_name)).load())
                doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")

        return docs

    def __chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk the loaded documents using the specified text splitter.

        :param docs: List[Document]
        :return: List[Document]
        """
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        """
        Load, chunk, and create a VectorDB with embeddings, and save it.

        :return:
            Chroma: The created VectorDB.
        """
        docs = self.__load_all_documents()
        chunked_documents = self.__chunk_documents(docs)

        print("Preparing vectordb...")

        if len(chunked_documents) > 0:
            vectordb = Chroma.from_documents(chunked_documents,
                                             self.embedding_function,
                                             persist_directory=self.persist_directory)

            print("VectorDB is created and saved.")
            print("Number of vectors in vectordb:", vectordb._collection.count(), "\n\n")

            return vectordb
        else:
            print("VectorDB can not be created, number of chunked_documents is 0!")