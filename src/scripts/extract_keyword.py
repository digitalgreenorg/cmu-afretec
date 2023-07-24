# This script extracts keywords from dataset by getting values from columns with unique values < 5
# and by extracting keywords from generated summary/ description

# Imports
import os
from dotenv import load_dotenv
from typing import List

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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_tags(description: str, dataset_path: str) -> List:

    llm_2 = OpenAI(temperature=0)

    if dataset_path.endswith('.csv'):
        data_file = pd.read_csv(dataset_path)
    
    elif dataset_path.endswith('.xlsx'):
        data_file = pd.read_csv(dataset_path)
    
    string_data = data_file.select_dtypes(include=["object"])
    data_unique_num = string_data.nunique()
    select_data = data_unique_num[data_unique_num < 7]

    unique_values = []

    select_columns = select_data.keys()

    for column in select_columns:
        unique_values.extend(data_file[column].unique())

    unique_values = [ value for value in unique_values if isinstance(value, str) and len(value) < 16]

    if  len(unique_values) != 0:
        f_query = " Pick relevant Agricultural keywords from this string "
        context = f"{' '.join(unique_values)}"
        tags = llm_2(context + f_query).replace('\n', '').split(', ')

    return tags

if __name__ == '__main__':
    extract_tags()