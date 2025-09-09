# Janus: A Conversational Document Intelligence Assistant

Janus is a private, locally-run AI assistant capable of having intelligent, contextual conversations about your documents. This project implements a complete Retrieval-Augmented Generation (RAG) pipeline from the ground up, demonstrating a robust and practical application of modern AI engineering principles.

---
## Why Janus is Private & Secure

This application is designed with privacy as a core feature. All components run locally on your machine, ensuring your documents and conversations are never exposed to external services.
* **Local Compute**: All processing, from document ingestion to LLM inference, happens on your own hardware.
* **No API Keys**: Janus uses a local, quantized GGUF model and open-source embedding models, requiring no external API keys or paid services.
* **Offline Functionality**: Once the models are downloaded, the entire system can run without an internet connection.

---
## How It Works: The RAG Pipeline

Janus uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers. The process can be broken down into five steps:

1.  **Ingest**: The `ingest.py` script loads PDF documents from the `documents/` folder.
2.  **Chunk**: The text is divided into smaller, semantically coherent chunks.
3.  **Embed**: Each chunk is converted into a numerical vector (embedding) using a local Sentence-Transformers model.
4.  **Store**: The embeddings are stored in a local FAISS vector database, creating a searchable knowledge base.
5.  **Retrieve & Generate**: When you ask a question, the system retrieves the most relevant chunks from the database and feeds them, along with your conversation history, to the local LLM to generate a final, synthesized answer.

---
## Key Features

* **100% Private**: Runs entirely offline on your local machine.
* **Conversational Memory**: Remembers the context of the chat for natural follow-up questions.
* **Local LLM Deployment**: Powered by a quantized Mistral-7B GGUF model.
* **End-to-End RAG Pipeline**: Implements the full ingestion and retrieval workflow.
* **Polished CLI**: A professional and user-friendly command-line interface with clear instructions and guardrails.

---
## Getting Started

### Prerequisites
* Python 3.10+
* Git

### Quickstart Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Milanseban/janus.git](https://github.com/Milanseban/janus.git)
    cd janus
    ```
2.  **Set up the environment and install dependencies:**
    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download the LLM:**
    Download the [Mistral-7B-Instruct-v0.2.Q4_K_M.gguf model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf). Then, create a `models` folder in the project directory and place the downloaded file inside it.

4.  **Add your documents:**
    Place the PDF files you want to analyze inside the `documents/` folder.

5.  **Build the knowledge base:**
    Run the ingestion script to process your documents. This only needs to be done once per set of documents.
    ```bash
    python ingest.py
    ```
6.  **Run the chatbot:**
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

---
## Screenshots

<img width="908" height="488" alt="janus" src="https://github.com/user-attachments/assets/5f567e5c-6d15-4f86-bb1e-1e79df11d5eb" />

