from src.utils.load_config import LoadConfig
from dotenv import load_dotenv
from openai import OpenAI
import os

CONFIG = LoadConfig()

load_dotenv()

OPENAI_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_KEY)


def get_response(system_role, user_prompt):
    response = client.chat.completions.create(
        model=CONFIG.model_name,
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content
