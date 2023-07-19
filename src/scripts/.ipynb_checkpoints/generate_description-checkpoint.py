# 
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# LangChain imports
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredExcelLoader, UnstructuredFileLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate


api_key = os.getenv("OPENAI_API_KEY")
# Display all columns from dataset
pd.set_option('display.max_columns', None)

# function to engineer prompt with column names only
def create_prompt_col(col):
    prompt = PromptTemplate(
        input_variables = ["columns"],
        template = "Explain this dataset based on {columns}"
    )
    query = prompt.format(columns = list(col))
    return query

# function for prompt engineering
# makes prompt from column names and 5 first rows
def create_prompt(col, row):
    prompt = PromptTemplate(
    input_variables=["columns", "rows"],
    template = "Generate a summary about this dataset based on {columns} and {rows} information"
    )
    query = prompt.format(columns = list(col), rows = row)
    return query

# function to generate description with pandas agent
def generate_desc_pd_agent(df, question):
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0),
                                         df,
                                         verbose=True
                                         )
    description = agent.run(question)
    return description

# function to generate description from chatGPT model
def gen_desc_gpt(query):
    description = llm(query)
    return description

# function to generaate description with Retrieval QA
def gen_desc_rQA(file, api_k, llm, query):
    df = pd.read_excel(file)
    query_gpt = create_prompt_col(df.columns)
    loader = UnstructuredFileLoader(file)
    doc = loader.load()
    
    # split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    docs = text_splitter.split_documents(doc)
    
    total_num_char = sum([len(x.page_content) for x in docs])
    # print(f"Now you have {len(docs[:4])} documents that have an average of {total_num_char / len(docs):,.0f} characters")
    # embeddings
    # creating vectors db
    embeddings = OpenAIEmbeddings(openai_api_key=api_k)
    docsearch = FAISS.from_documents(docs[:4], embeddings)
    
    # Create retrieval QA engine
    qa = RetrievalQA.from_chain_type(llm = llm,
                                     chain_type = "stuff",
                                     retriever = docsearch.as_retriever(),
                                     verbose = True)
    
    try:                     
        description = qa.run(query)
    except Exception:
        print("Error message:")
        description = gen_desc_gpt(query_gpt)
    finally:
        return description