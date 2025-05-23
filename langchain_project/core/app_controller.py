import os
import sys
from dotenv import load_dotenv

# --- Add project root to sys.path for simpler imports ---
# This allows us to use absolute imports like `from utils.prompt_loader import ...`
# irrespective of where the script is run from within the project.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
# This will load variables from a .env file in the PROJECT_ROOT if it exists.
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)
# --- End sys.path and dotenv modification ---

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Import from our framework modules
from utils.prompt_loader import load_prompt_template
from embeddings.embedding_manager import EmbeddingManager
from chains.function_calling_chain import create_function_calling_agent_executor
from langchain_openai import OpenAI # For the simple joke chain

def run_simple_joke_chain(api_key, topic="AI"):
    print(f"\n--- Running Simple Joke Chain (using loaded prompt for topic: '{topic}') ---")
    try:
        llm = OpenAI(openai_api_key=api_key, temperature=0.7)
        joke_prompt_template = load_prompt_template("joke_prompt", prompts_dir="prompts")
        
        # Basic chain (not using LLMChain from langchain.chains explicitly here for simplicity,
        # but PromptTemplate + LLM is the core of it)
        formatted_prompt = joke_prompt_template.format(topic=topic)
        print(f"Formatted Prompt: {formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        print(f"LLM Response (Joke):\n{response.strip()}")
    except FileNotFoundError:
        print("ERROR: joke_prompt.json not found. Make sure it's in the 'prompts' directory.")
    except Exception as e:
        print(f"Error in simple joke chain: {e}")

def run_embedding_demo(embedding_manager):
    print("\n--- Running Embedding Manager & FAISS Demo ---")
    sample_texts_for_vectorstore = [
        "Langchain is a powerful framework for LLM application development.",
        "Embeddings convert text into numerical vectors.",
        "FAISS allows for efficient similarity search of embeddings.",
        "The capital of France is Paris.",
        "Apples are a type of fruit."
    ]
    documents = [Document(page_content=text) for text in sample_texts_for_vectorstore]

    try:
        print("Generating embeddings for sample documents...")
        # The embedding_model from the manager is used by FAISS
        vector_store = FAISS.from_documents(documents, embedding_manager.embedding_model)
        print("In-memory FAISS vector store created.")

        query = input("Enter a query for similarity search (e.g., 'What is Langchain?'): ")
        if not query:
            print("No query entered. Skipping similarity search.")
            return

        similar_docs = vector_store.similarity_search(query, k=2)
        print(f"\nFound {len(similar_docs)} documents similar to '{query}':")
        for i, doc in enumerate(similar_docs):
            print(f"  {i+1}. '{doc.page_content}'")
    except Exception as e:
        print(f"Error in embedding demo: {e}")

def run_function_calling_agent_demo(agent_executor):
    print("\n--- Running Function Calling Agent Demo ---")
    query = input("Enter a query for the function-calling agent (e.g., 'What's the weather in London?' or 'What is 5 * 13?'): ")
    if not query:
        print("No query entered. Skipping agent demo.")
        return
    try:
        response = agent_executor.invoke({"input": query, "chat_history": []})
        print(f"\nAgent Response:\n{response.get('output')}")
    except Exception as e:
        print(f"Error in function calling agent demo: {e}")

def main():
    print("--- Langchain Advanced Framework Demo Controller ---")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("CRITICAL ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this application.")
        print("You can get an API key from https://platform.openai.com/account/api-keys")
        print("The application will attempt to run parts that don't require the key, but most will fail.")
        # Depending on strictness, you might exit() here.
        # For demo purposes, we'll allow it to proceed and fail on specific parts.

    # 1. Simple Joke Chain (uses basic LLM and loaded prompt)
    if openai_api_key:
        joke_topic = input("Enter a topic for a joke (e.g., 'programmers', 'cats', default: AI): ") or "AI"
        run_simple_joke_chain(openai_api_key, topic=joke_topic)
    else:
        print("\nSKIPPING: Simple Joke Chain (requires OPENAI_API_KEY)")

    # 2. Embedding Manager and FAISS Demo
    embedding_manager = None
    if openai_api_key:
        try:
            embedding_manager = EmbeddingManager(api_key=openai_api_key)
            print("\nEmbeddingManager initialized successfully.")
            run_embedding_demo(embedding_manager)
        except Exception as e:
            print(f"Could not initialize or run embedding demo: {e}")
    else:
        print("\nSKIPPING: Embedding Manager & FAISS Demo (requires OPENAI_API_KEY)")
    
    # 3. Function Calling Agent Demo
    if openai_api_key:
        try:
            agent_executor = create_function_calling_agent_executor(api_key=openai_api_key)
            print("\nFunction-calling agent executor created successfully.")
            run_function_calling_agent_demo(agent_executor)
        except Exception as e:
            print(f"Could not initialize or run function calling agent: {e}")
    else:
        print("\nSKIPPING: Function Calling Agent Demo (requires OPENAI_API_KEY)")

    print("\n--- Demo Controller Finished ---")

if __name__ == "__main__":
    main()
