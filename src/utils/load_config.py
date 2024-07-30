from huggingface_hub.hf_api import HfFolder
from dotenv import load_dotenv
from pyprojroot import here
from typing import List
import shutil
import yaml
import os

load_dotenv()


class LoadConfig:
    """
    A class for loading configuration settings and managing directories.

    This class loads various configuration settings from the 'app_config.yml' file,
    including language model (LLM) configs, retrieval configs, summarizer configs
    and memory configs. It also sets up OpenAI API credentials and performs
    directory-related operations such as creating and removing directories.

    ...

    Attributes:
        model_name : str
            The language model engine specified in the configuration.
        system_message : str
            The role of the language model system specified in the configuration.
        persist_directory: str
            The path to the persist directory where data is stored.
        custom_persist_directory : str
            The path to the custom persist directory.
        data_directory : str
            The path to the data directory.
        k : int
            The value of 'k' specified in the retrieval configuration.
        embedding_model_config : str
            The engine specified in the embedding model configuration.
        chunk_size : int
            The chunk size specified in the splitter configuration.
        chunk_overlap : int
            The chunk overlap specified in the splitter configuration.
        max_final_token : int
            The maximum number of final tokens specified in the summarizer configuration.
        token_threshold: float
            The token threshold specified in the summarizer configuration.
        summarizer_llm_system_role : str
            The role of the summarizer language model system specified in the configuration.
        temperature : float
            The temperature specified in the LLM configuration.
        number_of_q_a_pairs : int
            The number of question-answer pairs specified in the memory configuration.

    Methods:
        load_huggingface_cfg():
            Load Hugging Face configuration settings.
        create_directory(directory_path):
            Create a directory if it does not exist.
        remove_directory(directory_path):
            Removes the specified directory.
    """

    # Class variable to track initialization
    _initialized = False

    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Data directories
        self.data_directory = app_config["directories"]["data_directory"]
        self.persist_directory = str(here(app_config["directories"]["persist_directory"]))
        self.custom_persist_directory = str(here(app_config["directories"]["custom_persist_directory"]))

        # OpenAI config
        self.model_name = app_config["openai_config"]["model_name"]
        self.system_message = app_config["openai_config"]["system_message"]
        self.user_message = app_config["openai_config"]["user_message"]

        # LLM config
        self.max_new_tokens = app_config["llm_config"]["max_new_tokens"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.llm_template = app_config["llm_config"]["llm_template"]

        # Embedding config
        self.embedding_model_config = app_config["embedding_model_config"]["engine"]

        # Retrieval configs
        self.k = app_config["retrieval_config"]["k"]
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        # Summarizer config
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.final_summarizer_llm_system_role = app_config["summarizer_config"]["final_summarizer_llm_system_role"]

        # Memory
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

        # Flask endpoint
        self.flask_app_endpoint = app_config["serve"]["flask_app_endpoint"]

        # Load HuggingFace credentials
        self.load_huggingface_cfg()

        if not LoadConfig._initialized:
            # Clean up the upload doc vectordb if it exists
            self.remove_directory(self.custom_persist_directory)
            self.create_directory([self.data_directory, self.persist_directory, self.custom_persist_directory])

            # Set initialization flag to True
            LoadConfig._initialized = True

    @staticmethod
    def load_huggingface_cfg():
        """
        Load Hugging Face configuration settings.
        """
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if hf_token:
            HfFolder.save_token(hf_token)
        else:
            print("Hugging Face API token not found. Please set the HUGGINGFACE_API_TOKEN environment variable.")

    def create_directory(self, dirs: List[str]):
        """
        Creates a directory if it does not exist.

        :param:
            dirs List[str]: The list of directories paths to be created
        """
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"The directory '{directory}' has been created.")

    def remove_directory(self, directory: str):
        """
        Removes the specified directory.

        :param:
            dir str: The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        :return:
            None
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"The directory '{directory}' has been successfully removed.")
