
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


embeddings_model_name =  "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


def ingest(file_path, persist_directory):
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    chunk_size = 1000
    chunk_overlap = 200
    persist_directory = persist_directory
    embeddings_model_name = 'all-MiniLM-L6-v2'

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    texts = text_splitter.split_documents(pages)
    file_name = 'physics'
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, collection_name=file_name)
    print(f"Ingestion complete! You can now run chatbot.py to query your documents")






