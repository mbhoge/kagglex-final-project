import os
from dotenv import load_dotenv
import openai
import langchain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

# Define NLP class
class GenAILearningPathIndex:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.vectorizer = None
        self.model = None

    def load_data(self):
        # Load your dataset (e.g., CSV, JSON, etc.)
        self.data = pd.read_csv(self.data_path)


if __name__=='__main__':
    load_dotenv()  # Load .env file
    # Rest of your code goes here
    my_key = os.getenv("OPENAI_API_KEY")
    print(f"Key is : {my_key}")
    llm = OpenAI(temperature=0.1, openai_api_key=my_key)
    text = "what is AI?"
    print(llm(text))