import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_response(user_text_emotion,user_text,agent_emotion):
    response = client.responses.create(
    model="gpt-5-nano",
    instructions="Eres un agente llamada Mia empatica, dulce y tierna pero no sumisa que genera respuestas solo en español sin emojis",
    input="Are semicolons optional in JavaScript?",
    )
    return response.output_text

