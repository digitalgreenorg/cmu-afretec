# Imports
import os
import sys
from dotenv import load_dotenv

import pandas as pd

# LangChain imports
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredExcelLoader, UnstructuredFileLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from extract_keyword import extract_tags

load_dotenv('../../.env.production')

if len(sys.argv) < 2:
    print("You have to specify the name of the dataset \nExiting...")
    exit()

elif len(sys.argv) > 2:
    print("Function takes only one argument (name of dataset) but more than one argument found \nExiting...")
    exit()

else:
    file_path = f"../datasets/{sys.argv[1]}"
    if os.path.exists(file_path):
        print(f"The dataset {file_path} was found \nProceeding...")
    else:
        print(f"The path {file_path} to the the dataset does not exist \nExiting")
        exit()

api_key = os.getenv("OPENAI_API_KEY")
# Display all columns from dataset
pd.set_option('display.max_columns', None)
llm = OpenAI(temperature=0, openai_api_key = api_key)

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
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    description = agent.run(question)
    return description

# function to generate description from chatGPT model
def gen_desc_gpt(query):
    description = llm(query)
    return description

# function to generaate description with Retrieval QA
def gen_desc_rQA(dataset_path: str, generate_tags: bool = True) -> str:

    if dataset_path.endswith('.csv'):
        data_file = pd.read_csv(dataset_path)
    
    elif dataset_path.endswith('.xlsx'):
        data_file = pd.read_excel(dataset_path)
    
    else:
        print("Please provide a valid dataset in the csv or excel format")
        return
    
    query = "Give an elaborate description of the dataset"
    loader = UnstructuredFileLoader(dataset_path)
    doc = loader.load()
    
    # split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)
    
    total_num_char = sum([len(x.page_content) for x in docs])
    # print(f"Now you have {len(docs[:4])} documents that have an average of {total_num_char / len(docs):,.0f} characters")
    # embeddings
    # creating vectors db
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docsearch = FAISS.from_documents(docs[:4], embeddings)
    
    # Create retrieval QA engine
    qa = RetrievalQA.from_chain_type(llm = llm,
                                     chain_type = "stuff",
                                     retriever = docsearch.as_retriever(),
                                     verbose = True
    )
    
    try:                     
        description = qa.run(query)
    except Exception:
        query_gpt = create_prompt_col(data_file.columns)
        description = gen_desc_gpt(query_gpt)
    finally:
        description = description
    
    if generate_tags:
        tags = extract_tags(description, file_path)
    
    return (description, tags)

if __name__ == '__main__':
    description, tags = gen_desc_rQA(file_path, api_key)
    print("\nDataset Description:")
    print(f"{description}\n")
    print("\nSuggested Dataset Tags:")
    print(tags)