import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCUMENTS_PATH = "documents/"
VECTORSTORE_PATH = "vectorstore/"

def load_documents(directory_path):
    print(f"Loading documents from {directory_path}...")
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} documents.")
    return documents

def chunk_documents(documents):
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Successfully chunked documents into {len(chunked_docs)} chunks.")
    return chunked_docs

def create_and_save_vectorstore(chunks):
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store created and saved to {VECTORSTORE_PATH}")

if __name__ == "__main__":
    loaded_docs = load_documents(DOCUMENTS_PATH)
    chunked_docs = chunk_documents(loaded_docs)
    create_and_save_vectorstore(chunked_docs)