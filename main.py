from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ctransformers import AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from rich.console import Console
from rich.panel import Panel

# Define constants
VECTORSTORE_PATH = "vectorstore/"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

ANSWER_PROMPT_TEMPLATE = """You are a helpful and precise assistant for answering questions based on provided text.
To answer the question, first, identify the core concepts in the provided context. Second, reason step-by-step if the context is describing a problem of an old architecture or a feature/limitation of the new Transformer architecture. Third, synthesize the information to provide a concise answer.
If you don't know the answer from the context, just say that you don't know.

Context: {context}

Question: {question}
Helpful Answer:"""


def load_llm(model_path):
    """Loads the local GGUF model directly."""
    # This function does not need console passed to it
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=model_path,
        model_type="mistral",
        max_new_tokens=512,
        temperature=0.7,
        context_length=2048
    )
    return llm


class LLMRunnable:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, prompt):
        return self.llm(prompt.to_string())


def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_output(answer, source_documents, console):
    """Formats the RAG chain's result for a polished display using rich."""
    console.print("\n\n[bold green]--- Janus Answer ---[/bold green]")
    console.print(answer)

    if source_documents:
        console.print("\n[bold green]--- Sources ---[/bold green]")
        unique_sources = set()
        for doc in source_documents:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            unique_sources.add(f"- Source: {source}, Page: {page}")

        for source_info in sorted(list(unique_sources)):
            console.print(source_info)
    console.print("[bold green]-------------------[/bold green]\n")


def main():
    """Main function for the interactive chatbot."""
    console = Console()

    console.print("Loading vector store...", style="bold yellow")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    console.print("Vector store and retriever loaded successfully.", style="bold green")

    console.print("Loading LLM...", style="bold yellow")
    llm = load_llm(MODEL_PATH)
    runnable_llm = LLMRunnable(llm)
    console.print("LLM loaded successfully.", style="bold green")

    # Define prompts and memory
    condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)
    answer_prompt = PromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)
    memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

    # Define chains
    standalone_question_chain = {"standalone_question": {"question": lambda x: x["question"], "chat_history": lambda
        x: memory.chat_memory.messages, } | condense_question_prompt | runnable_llm | StrOutputParser(), }
    retrieved_documents_chain = {"docs": itemgetter("standalone_question") | retriever,
                                 "question": itemgetter("standalone_question"), }
    final_inputs_chain = {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], }
    answer_chain = final_inputs_chain | answer_prompt | runnable_llm | StrOutputParser()

    console.print(Panel(
        "[bold cyan]Janus is ready.[/bold cyan]\nYou can now ask questions about your documents.\nType '[bold red]exit[/bold red]' to end the conversation.",
        title="[bold]Welcome to Janus[/bold]",
        subtitle="[italic]Your Document Intelligence Assistant[/italic]"
    ))

    while True:
        question = console.input("[bold]Your Question: [/bold]")
        if question.lower() == 'exit':
            console.print("[bold yellow]Exiting Janus. Goodbye![/bold yellow]")
            break

        # --- This is the key change: Add the status indicator ---
        with console.status("[bold yellow]Janus is thinking...[/bold yellow]", spinner="dots"):
            # Generate the standalone question
            standalone_question = standalone_question_chain["standalone_question"].invoke({"question": question})

            # Retrieve relevant documents
            relevant_docs = retriever.invoke(standalone_question)

            # Get the final answer
            answer = answer_chain.invoke({"question": standalone_question, "docs": relevant_docs})

        # Save context and display the formatted output
        memory.save_context({"question": question}, {"answer": answer})
        format_output(answer, relevant_docs, console)


if __name__ == "__main__":
    main()