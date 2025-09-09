import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rich.console import Console
from rich.panel import Panel

# Initialize Rich Console for better printing
console = Console()

# Define constants for file paths
DOCUMENTS_PATH = "documents/"
VECTORSTORE_PATH = "vectorstore/"


def load_documents(directory_path):
    """
    Loads all PDF documents from a given directory.
    """
    console.print(f"Loading documents from [cyan]{directory_path}[/cyan]...", style="bold yellow")
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    console.print(f"Successfully loaded {len(documents)} documents.", style="bold green")
    return documents


def chunk_documents(documents):
    """
    Splits the loaded documents into smaller chunks.
    """
    console.print("Chunking documents...", style="bold yellow")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunked_docs = text_splitter.split_documents(documents)
    console.print(f"Successfully chunked documents into {len(chunked_docs)} chunks.", style="bold green")
    return chunked_docs


def create_and_save_vectorstore(chunks):
    """
    Creates embeddings for the chunks and saves them to a FAISS vector store.
    """
    console.print("Creating embeddings and vector store...", style="bold yellow")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)
    console.print(f"Vector store created and saved to [cyan]{VECTORSTORE_PATH}[/cyan]", style="bold green")


# Main execution block
if __name__ == "__main__":
    # --- New Guardrail ---
    if not os.path.exists(DOCUMENTS_PATH) or not os.listdir(DOCUMENTS_PATH):
        console.print(Panel(
            f"[bold red]Error: Documents folder is empty![/bold red]\n\nThe '[bold cyan]{DOCUMENTS_PATH}[/bold cyan]' folder does not contain any files.\nPlease add your PDF documents to this folder before running the ingestion script.",
            title="[bold red]Setup Error[/bold red]",
            border_style="red"
        ))
    else:
        # Run the complete ingestion pipeline
        loaded_docs = load_documents(DOCUMENTS_PATH)
        chunked_docs = chunk_documents(loaded_docs)
        create_and_save_vectorstore(chunked_docs)