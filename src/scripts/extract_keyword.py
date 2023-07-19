# This script extracts keywords from dataset by getting values from columns with unique values < 5
# and by extracting keywords from generated summary/ description

# Imports
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

# method to extract keywords
def extract_keywords():
    pass