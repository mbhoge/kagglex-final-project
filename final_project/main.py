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

# https://discuss.streamlit.io/t/how-to-check-if-code-is-run-inside-streamlit-and-not-e-g-ipython/23439/7
def running_inside_streamlit():
    """
    Function to check whether python code is run within streamlit

    Returns
    -------
    use_streamlit : boolean
        True if code is run within streamlit, else False
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit


# Define GenAI class
class GenAILearningPathIndex:
    def __init__(self, faiss_vectorstore):
        load_dotenv()  # Load .env file
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.faiss_vectorstore = faiss_vectorstore

        prompt_template = \
            """
                Use the following template to answer the question at the end, from the Learning Path Index csv file.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Display the results in a table, results must contain a link for each line of the result.
                {context}
                Question: {question}
            """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
        self.chain_type_kwargs = {"prompt": PROMPT}

        self.llm = OpenAI(temperature=0.1, openai_api_key=self.openai_api_key)
       
    def get_query(self, query: str):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", 
            retriever=self.faiss_vectorstore.as_retriever(),
            chain_type_kwargs=self.chain_type_kwargs
        )
        return qa.run(query)

# Class definition ends here

def get_formatted_time(current_time = time.time()):
    return datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

@st.cache_data
def load_model():
    start_time = time.time()
    print(f"\nStarted loading model at {get_formatted_time(start_time)}")
    learningPathIndexModel = LoadLearningPathIndexModel()
    faiss_vectorstore = learningPathIndexModel.get_faiss_vector_store()
    end_time = time.time()
    print(f"Finished loading model at {get_formatted_time(end_time)}")
    print(f"Model took about {end_time - start_time} seconds) to load.")
    return faiss_vectorstore

if __name__=='__main__':
    faiss_vectorstore = load_model()
    genAIproject = GenAILearningPathIndex(faiss_vectorstore)

    if running_inside_streamlit():
        print("\nStreamlit environment detected.\n")
        query_from_stream_list = app()
        if query_from_stream_list:
            start_time = time.time()
            print(f"\nQuery processing start time: {get_formatted_time(start_time)}")
            answer = genAIproject.get_query(query_from_stream_list)
            end_time = time.time()
            print(f"\nQuery processing finish time: {get_formatted_time(end_time)}")
            print(f"\nAnswer (took about {end_time - start_time} seconds)")
            st.write(answer)
    else:
        print("\nCommand-line interactive environment detected.\n")
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            if query.strip() == "":
                continue

            # Get the answer from the chain
            start_time = time.time()
            print(f"\nQuery processing start time: {get_formatted_time(start_time)}")
            answer = genAIproject.get_query(query)
            end_time = time.time()

            # Print the result
            print("\n\n> Question:")
            print(query)
            print(f"\nQuery processing finish time: {get_formatted_time(end_time)}")
            print(f"\nAnswer (took about {end_time - start_time} seconds):")
            print(answer)
