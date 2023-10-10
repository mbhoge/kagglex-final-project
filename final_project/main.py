import os
from dotenv import load_dotenv
import openai
import langchain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA

# Define NLP class
class GenAILearningPathIndex:
    def __init__(self, data_path):
        load_dotenv()  # Load .env file
        # Rest of your code goes here
        self.my_key = os.getenv("OPENAI_API_KEY")
        self.data_path = data_path
        self.data = None
        self.vectorizer = None
        self.model = None
        self.text = None
        self.embeddings = None

    def load_data(self):
        # Load your dataset (e.g., CSV, JSON, etc.)
        loader = TextLoader(self.data_path)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.text = text_splitter.split_documents(document)
    
    def getllm(self):
        llm = OpenAI(temperature=0.1, openai_api_key=self.my_key)
        result = llm("what is AI?")
        print(result)
        
    def getembeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.my_key)
    
    def pinecone_init(self):
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="asia-southeast1-gcp-free")
        

if __name__=='__main__':
    data_path="C:\\Users\\Manish_Bhoge\\OneDrive - EPAM\\Tut\\Kaggle-Mentership-Program\\Project - GenAI\\kagglex-final-project\\final_project\\Learning_Pathway_Index_1.csv"
    GenAI_project = GenAILearningPathIndex(data_path)
    # print(f"Key is : {GenAI_project.my_key}")
    # GenAI_project.text = "what is AI?"
    GenAI_project.getllm()
    GenAI_project.load_data()
    GenAI_project.getembeddings()
    GenAI_project.pinecone_init()
    
    docsearch = Pinecone.from_documents(GenAI_project.text, GenAI_project.embeddings, index_name="genai-learning-path-index")
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    
    query = "Give me Machine Learning Course with 10 min duration"
    result = qa({"query":query})
    print(result)
    #print(len(text))
    
    