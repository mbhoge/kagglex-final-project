import os
from dotenv import load_dotenv
import openai
import langchain
from langchain.llms import OpenAI

load_dotenv()  # Load .env file

# Rest of your code goes here
my_key = os.getenv("OPENAI_API_KEY")
print(f"Key is : {my_key}")

llm = OpenAI(temperature=0.1, openai_api_key=my_key)
text = "what is AI?"
print(llm(text))