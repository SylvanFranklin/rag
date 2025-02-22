from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os
import shutil

load_dotenv("/Users/sylvanfranklin/documents/projects/rag/secrets.env")
openai.api_key = os.environ["OPENAI_API_KEY"]
DATA_PATH = os.environ["DATA_PATH"]
CHROMA_PATH = os.environ["CHROMA_PATH"]


def load():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents


def split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len, add_start_index=True
    )

    chunks = splitter.split_documents(documents)
    return chunks


def load_into_database(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)


def generate_database():
    docs = load()
    chunks = split(docs)
    load_into_database(chunks)
