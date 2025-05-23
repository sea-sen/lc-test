import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # Keep if used in __main__
from langchain.docstore.document import Document # Keep if used in __main__
from dotenv import load_dotenv

# Load .env file from project root.
# This assumes the module is part of `langchain_project`.
PROJECT_ROOT_EMBEDDINGS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path_embeddings = os.path.join(PROJECT_ROOT_EMBEDDINGS, '.env')
load_dotenv(dotenv_path=dotenv_path_embeddings)


class EmbeddingManager:
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-ada-002"):
        """
        Initializes the EmbeddingManager.
        Tries to load OPENAI_API_KEY from .env file or environment variables if api_key is not provided.
        """
        # load_dotenv() # Called globally at module load for simplicity here
        
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # This message might need adjustment if .env is the primary expected source
            raise ValueError("OPENAI_API_KEY not provided directly or found in .env file or environment variables.")
        
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key, model=model_name)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts."""
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings.")
        return self.embedding_model.embed_documents(texts)

    def get_embedding_for_query(self, text: str) -> list[float]:
        """Generates embedding for a single query text."""
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string.")
        return self.embedding_model.embed_query(text)

# Example Usage and Simple Vector Store Demo
if __name__ == '__main__':
    print("Running EmbeddingManager and FAISS Demo...")

    # IMPORTANT: Ensure OPENAI_API_KEY is set in your environment variables or .env file
    # or pass it directly: EmbeddingManager(api_key="your_key_here")
    try:
        # load_dotenv() is called at module level, so EmbeddingManager should find the key
        manager = EmbeddingManager() 
        print("EmbeddingManager initialized.")
    except ValueError as e:
        print(f"Error initializing EmbeddingManager: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable or in your .env file.")
        print("Skipping further demo.")
        exit()

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Langchain is a framework for developing applications powered by language models.",
        "Paris is the capital of France.",
        "Embeddings are numerical representations of text."
    ]
    
    try:
        print(f"\nGenerating embeddings for {len(sample_texts)} texts...")
        embeddings = manager.get_embeddings(sample_texts)
        print(f"Successfully generated {len(embeddings)} embeddings.")
        print(f"Dimension of the first embedding: {len(embeddings[0])}")

        print("\nGenerating embedding for a single query 'Hello world':")
        query_embedding = manager.get_embedding_for_query("Hello world")
        print(f"Dimension of query embedding: {len(query_embedding)}")

        # Simple FAISS Vector Store Example
        print("\n--- FAISS Vector Store Demo ---")
        # Convert texts to Langchain Document objects for FAISS
        documents = [Document(page_content=text) for text in sample_texts]
        
        print("Creating FAISS index from documents...")
        # FAISS.from_texts uses the same embedding_model from the manager
        vector_store = FAISS.from_documents(documents, manager.embedding_model)
        print("FAISS index created successfully.")

        query = "What is Langchain?"
        print(f"\nPerforming similarity search for query: '{query}'")
        
        # The retriever will embed the query using manager.embedding_model
        # and find similar documents in the FAISS index.
        similar_docs = vector_store.similarity_search(query, k=2)
        
        print(f"Found {len(similar_docs)} similar documents:")
        for i, doc in enumerate(similar_docs):
            print(f"  {i+1}. Content: \"{doc.page_content}\"")
            # If you want to see scores (distances), you can use similarity_search_with_score
            # similar_docs_with_score = vector_store.similarity_search_with_score(query, k=2)
            # print(f"      Score: {similar_docs_with_score[i][1]}")


    except Exception as e:
        print(f"An error occurred during the demo: {e}")
