import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class GemmaClient:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("GEMMA_API_KEY"),
            base_url=os.getenv("GEMMA_BASE_URL")
        )

        self.model = os.getenv(
            "GEMMA_MODEL"
        )


    def generate(self, prompt):

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1
        )


        return response.choices[0].message.content
