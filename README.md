# Janus: A Conversational Document Intelligence Assistant

Janus is a private, locally-run AI assistant capable of having intelligent, contextual conversations about your documents. This project implements a complete Retrieval-Augmented Generation (RAG) pipeline from the ground up, demonstrating a robust and practical application of modern AI engineering principles.

---
## Overview

The primary goal of this project is to create a secure and reliable tool for document analysis. By leveraging a local Large Language Model (LLM) and a vector database, Janus can ingest, understand, and answer questions about PDF documents without any data ever leaving the user's machine. The system is designed with conversational memory, allowing for natural, follow-up questions that build upon the previous context.

---
## Key Features

* **100% Private**: All document processing and language model inference occur locally. Your documents and conversations are never exposed to external services.
* **Conversational Memory**: The system maintains the context of the conversation, enabling it to understand and accurately answer follow-up questions.
* **Local LLM Deployment**: Powered by a quantized Mistral-7B GGUF model running efficiently on a CPU via the `ctransformers` library.
* **End-to-End RAG Pipeline**: Implements the full lifecycle of a RAG system:
    * **Ingestion**: Loads and parses PDF documents.
    * **Indexing**: Chunks text, creates embeddings, and stores them in a FAISS vector store.
    * **Retrieval & Generation**: Retrieves relevant context and synthesizes factual, grounded answers.
* **Polished Command-Line Interface**: A professional and user-friendly CLI built with the `rich` library provides a clean and interactive user experience.

---
## How to Run

### Prerequisites
* Python 3.10+
* Git

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Milanseban/janus.git](https://github.com/YOUR_USERNAME/janus.git)
    cd janus
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .\.venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the LLM:**
    Download the [Mistral-7B-Instruct-v0.2.Q4_K_M.gguf model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) and place it in a `models/` folder within the project directory.

### Usage

1.  **Add your documents:**
    Place the PDF files you want to analyze inside the `documents/` folder.

2.  **Build the knowledge base:**
    Run the ingestion script to process your documents. This only needs to be done once per set of documents.
    ```bash
    python ingest.py
    ```
3.  **Start the chatbot:**
    Run the main application to begin your conversational session.
    ```bash
    python main.py
    ```
---
## Technologies Used

* **Core Logic**: Python
* **AI Framework**: LangChain
* **LLM**: Mistral-7B (GGUF) via `ctransformers`
* **Embeddings**: `sentence-transformers`
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **CLI**: `rich`