import os
from dotenv import load_dotenv
from datetime import datetime
import time
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from interface import app
from langchain.prompts import PromptTemplate
import streamlit as st
from tqdm import tqdm

class LoadLearningPathIndexModel:
    # Initialize the Class
    def __init__(self):
        load_dotenv()  # Load .env file
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.data_path = os.path.join(os.getcwd(), "Learning_Pathway_Index.csv")
        self.text = None
        self.embeddings = None
        self.faiss_vectorstore = None

        # Load the data
        self.load_data()
        # Get the embeddings
        self.getembeddings()
        self.create_faiss_vectorstore()
           
    def load_data(self):
        # Load your dataset (e.g., CSV, JSON, etc.)
        loader = TextLoader(self.data_path)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        self.text = text_splitter.split_documents(document)
        
    def getembeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, request_timeout=60)
        
    def create_faiss_vectorstore(self):
        vectorstore = FAISS.from_documents(
            self.text, self.embeddings
        )
        vectorstore.save_local("faiss_learning_path_index")
        self.faiss_vectorstore = FAISS.load_local(
            "faiss_learning_path_index", self.embeddings
        )

    def get_faiss_vector_store(self):
        return self.faiss_vectorstore


# Define GenAI class
class GenAILearningPathIndex:
    def __init__(self, faiss_vectorstore):
        load_dotenv()  # Load .env file
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.faiss_vectorstore = faiss_vectorstore

        prompt_template = \
            """
                Use the following template to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Display the results in a tabular form, results must contain a link for each line of the result.
                {context}
                Question: {question}
            """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
        self.chain_type_kwargs = {"prompt": PROMPT}

        self.llm = OpenAI(temperature=0.1, openai_api_key=GenAILearningPathIndex.openai_api_key)
       
    def get_query(self, query: str):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", 
            retriever=self.faiss_vectorstore.as_retriever(),
            chain_type_kwargs=self.chain_type_kwargs
        )
        return qa.run(query)

# Class definition ends here

faiss_vectorstore = None
if __name__=='__main__':
    # var = app()

    # Initialize the Class

    if not faiss_vectorstore:
        learningPathIndexModel = LoadLearningPathIndexModel()
        faiss_vectorstore = learningPathIndexModel.get_faiss_vector_store()
    genAIproject = GenAILearningPathIndex(faiss_vectorstore)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        print(f"\nStart time: {datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')}")
        response = genAIproject.get_query(query)
        answer, docs = response['result'], []
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\nEnd time: {datetime.utcfromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAnswer (took about {end - start} seconds):")
        print(answer)


    # st.write(res)
    # st.write(datetime.now())