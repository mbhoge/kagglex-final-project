import os
from dotenv import load_dotenv
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from interface import app
import streamlit as st
       
# Define GenAI class
class GenAILearningPathIndex:
    def __init__(self, data_path):
        load_dotenv()  # Load .env file
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.data_path = data_path
        self.data = None
        self.vectorizer = None
        self.model = None
        self.text = None
        self.embeddings = None
        self.faiss_vectorstore = None

    def load_data(self):
        # Load your dataset (e.g., CSV, JSON, etc.)
        loader = TextLoader(self.data_path)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        self.text = text_splitter.split_documents(document)
    
    def getllm(self):
        llm = OpenAI(temperature=0.1, openai_api_key=self.openai_api_key)
        
    def getembeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
    
    def pinecone_init(self):
        pinecone.init(api_key=self.pinecone_api_key, environment="asia-southeast1-gcp-free")
        
    def create_index(self):
        docsearch = Pinecone.from_documents(self.text, self.embeddings, index_name="genai-learning-path-index")
        return docsearch 
    
    def faiss_index(self):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(self.text, self.embeddings)
        vectorstore.save_local("faiss_learning_path_index")
        faiss_vectorstore = FAISS.load_local("faiss_learning_path_index", self.embeddings)
        self.faiss_vectorstore = faiss_vectorstore     
# Class definition ends here

if __name__=='__main__':
    var = app()
    # st.write(f"The stored variable is: {var}")
    
    # Setting up the project
    current_directory = os.getcwd()
    data_path = current_directory + "\\final_project\\Learning_Pathway_Index.csv"
    # Initialize the Class
    GenAI_project = GenAILearningPathIndex(data_path)
    # Load the data
    GenAI_project.load_data()
    # Get the embeddings
    GenAI_project.getembeddings()
    # Initialize pinecone
    GenAI_project.pinecone_init()
    # Create the pincone index
    GenAI_project.create_index()
    # Create the QA model for pinecone
    # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    # Test the pinecone model
    # query = "Give me Machine Learning Course with 10 min duration. Are there any similar courses on coursera?"
    # result = qa({"query": query})
    # print(result)
    GenAI_project.faiss_index()
    # Create the QA model for pinecone
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=GenAI_project.faiss_vectorstore.as_retriever())
    # Test the FAISS model
    res = qa.run("Deep Learning Courses")
    st.write(res)
    #print(res)
    # comment out the below code to run the streamlit app       
    
    # llm = OpenAI(model="text-davinci-003", temperature=0.9)
    # prompt = PromptTemplate(
    #    input_variables=["user_input"],
    #    template="{user_input}. Can you also give ?",
    # )

    # chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    # print(chain.run("eco-friendly water bottles"))
    