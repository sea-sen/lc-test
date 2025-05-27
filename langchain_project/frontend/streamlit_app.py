import streamlit as st
import os
import sys

# --- Add project root to sys.path for simpler imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End sys.path modification ---

# Load environment variables from .env file, primarily for OPENAI_API_KEY
from dotenv import load_dotenv
import json # For displaying raw JSON
from utils.prompt_loader import load_prompt_template
from embeddings.embedding_manager import EmbeddingManager
from langchain.vectorstores import FAISS 
from langchain.docstore.document import Document
from chains.function_calling_chain import create_function_calling_agent_executor
# from langchain_core.messages import HumanMessage, AIMessage # If planning to manage chat history later

# Note: streamlit, os, sys are already imported.

dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Global variable to store API key status
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY_AVAILABLE = bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here")


def main_dashboard():
    st.title("Langchain Project Control Center")
    st.write("Welcome to the control center for your Langchain project.")
    st.write("Use the sidebar to navigate to different sections.")

    if not API_KEY_AVAILABLE:
        st.error("🔴 OPENAI_API_KEY is not configured or is invalid. Please set it in your `.env` file in the project root. Most features will be disabled.")
    else:
        st.success("✅ OPENAI_API_KEY is configured.")
        # st.write(f"Project Root: `{PROJECT_ROOT}`") # Optional: can be kept or removed

    st.subheader("Available Sections:")
    st.markdown(
        "- **Prompt Viewer:** Inspect and view the content of available prompt templates.\n"
        "- **Embedding Inspector:** Explore text embeddings, generate them for custom text, and (soon) perform similarity searches.\n"
        "- **LLM Interaction:** Configure and interact with LLMs and function-calling agents, observing their behavior and tool usage."
    )


# --- Placeholder functions for future pages ---
def prompt_viewer_page():
    st.header("Prompt Viewer")
    st.write("Inspect your Langchain prompt templates.")

    # Path to the prompts directory
    # Assumes PROJECT_ROOT is defined globally and correctly points to `langchain_project`
    prompts_dir = os.path.join(PROJECT_ROOT, "prompts")

    try:
        prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith(".json")]
    except FileNotFoundError:
        st.error(f"Prompts directory not found at: {prompts_dir}")
        st.write(f"Ensure the `prompts` directory exists in `{PROJECT_ROOT}`.")
        return
    except Exception as e:
        st.error(f"Error listing prompt files: {e}")
        return

    if not prompt_files:
        st.warning("No JSON prompt template files found in the 'prompts' directory.")
        st.write(f"Please add some .json prompt files to `{prompts_dir}`.")
        return

    # Remove .json extension for display in selectbox
    prompt_names = [os.path.splitext(f)[0] for f in prompt_files]
    
    selected_prompt_name = st.selectbox("Select a prompt template:", prompt_names)

    if selected_prompt_name:
        try:
            # We need to ensure utils.prompt_loader is imported.
            # It should be, due to the sys.path modification and global imports.
            # Let's assume `from utils.prompt_loader import load_prompt_template` is at the top.
            
            # The load_prompt_template function expects prompt_name without extension,
            # and prompts_dir relative to project root, which is fine as it constructs its own path.
            prompt_template = load_prompt_template(selected_prompt_name, prompts_dir="prompts")
            
            st.subheader(f"Details for: `{selected_prompt_name}`")
            
            st.markdown("#### Input Variables:")
            if prompt_template.input_variables:
                st.json(prompt_template.input_variables)
            else:
                st.write("None specified.")
            
            st.markdown("#### Template Content:")
            st.text_area("Template", prompt_template.template, height=200)
            
            # Display the raw JSON content from which it was loaded for full transparency
            # This requires re-reading the file, as load_prompt_template doesn't return raw JSON.
            raw_json_path = os.path.join(prompts_dir, selected_prompt_name + ".json")
            with open(raw_json_path, 'r') as f:
                raw_json_content = json.load(f) # Requires `import json`
            st.markdown("#### Raw JSON Source:")
            st.json(raw_json_content)

        except FileNotFoundError:
            st.error(f"Could not find the JSON file for prompt: {selected_prompt_name}.json")
        except Exception as e:
            st.error(f"Error loading or displaying prompt '{selected_prompt_name}': {e}")
            st.exception(e) # Provides more detailed traceback in Streamlit if needed


def embedding_inspector_page():
    st.header("Embedding Inspector")
    st.write("Inspect embedding models and generate embeddings for text.")

    if not API_KEY_AVAILABLE: # Assumes API_KEY_AVAILABLE is a global boolean
        st.error("🔴 Embedding Inspector requires an OPENAI_API_KEY to be configured in your .env file to function.")
        return

    try:
        # Assumes EmbeddingManager can be imported
        embedding_manager = EmbeddingManager(api_key=OPENAI_API_KEY) # Assumes OPENAI_API_KEY is global
        st.info(f"Using Embedding Model: `{embedding_manager.embedding_model.model}`")
    except Exception as e:
        st.error(f"Failed to initialize Embedding Manager: {e}")
        st.exception(e)
        return

    st.subheader("Generate Text Embedding")
    user_text = st.text_area("Enter text to generate embedding for:", "Hello, world!", height=100)

    if st.button("Generate Embedding"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                with st.spinner("Generating embedding..."):
                    embedding_vector = embedding_manager.get_embedding_for_query(user_text)
                
                st.success("Embedding generated successfully!")
                st.markdown(f"**Dimensionality:** `{len(embedding_vector)}`")
                st.markdown(f"**First 10 dimensions:**")
                st.code(str(embedding_vector[:10])) # Display first 10
                st.markdown("Full vector (first 100 dimensions shown below for brevity):")
                st.json(embedding_vector[:100])

            except Exception as e:
                st.error(f"Error generating embedding: {e}")
                st.exception(e)

    # --- Optional Stretch: Simple Similarity Search ---
    st.subheader("Simple Similarity Search (Optional Demo)")
    
    predefined_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Langchain is a framework for developing applications powered by language models.",
        "Paris is the capital of France.",
        "Large language models can understand and generate human-like text.",
        "Streamlit makes it easy to create web apps for machine learning projects."
    ]
    
    st.markdown("A small predefined set of documents will be indexed for this demo:")
    with st.expander("View predefined documents"):
        for i, text in enumerate(predefined_texts):
            st.markdown(f"- `{text}`")

    query_for_search = st.text_input("Enter text to search for similar documents:", "Tell me about LLMs")

    if st.button("Search Similar Documents"):
        if not query_for_search.strip():
            st.warning("Please enter some text for the search query.")
        elif not user_text.strip() and not query_for_search.strip(): # Edge case if main text area also empty
             st.warning("Please provide some text to embed or a search query.")
        else:
            try:
                with st.spinner("Performing similarity search..."):
                    # 1. Create Document objects
                    documents = [Document(page_content=text) for text in predefined_texts]
                    
                    # 2. Create FAISS index (in-memory)
                    # FAISS.from_documents uses the embedding_model from the manager
                    vector_store = FAISS.from_documents(documents, embedding_manager.embedding_model)
                    
                    # 3. Perform similarity search
                    similar_docs = vector_store.similarity_search(query_for_search, k=2)
                
                st.success("Similarity search complete!")
                if similar_docs:
                    st.markdown(f"Found **{len(similar_docs)}** similar document(s) to: `{query_for_search}`")
                    for i, doc in enumerate(similar_docs):
                        st.markdown(f"**{i+1}. Content:** `{doc.page_content}`")
                else:
                    st.info("No similar documents found.")

            except Exception as e:
                st.error(f"Error during similarity search: {e}")
                st.exception(e)


def llm_interaction_page():
    st.header("LLM Configuration & Interaction")
    st.write("Interact with the function-calling agent and observe its behavior.")

    if not API_KEY_AVAILABLE: # Assumes API_KEY_AVAILABLE is a global boolean
        st.error("🔴 LLM Interaction requires an OPENAI_API_KEY to be configured in your .env file to function.")
        return

    try:
        # Initialize the agent executor
        # Pass `return_intermediate_steps=True` to the AgentExecutor if not default,
        # or ensure the underlying agent creation supports it.
        # For `create_openai_functions_agent` followed by `AgentExecutor`,
        # the intermediate steps are captured if the agent is configured to provide them,
        # or if the AgentExecutor itself is set to return them.
        # The `AgentExecutor` by default should return intermediate steps if the agent provides them.
        # Let's assume the create_function_calling_agent_executor is already set up for verbose output
        # or we modify it if needed. For now, we'll rely on its existing structure.
        # The `verbose=True` in AgentExecutor prints to console, not directly capturable by Streamlit without context managers.
        # A better way is to get structured intermediate steps.
        # The standard output from AgentExecutor includes 'intermediate_steps'.

        agent_executor = create_function_calling_agent_executor(api_key=OPENAI_API_KEY)
        # The model name is somewhat hardcoded in `create_function_calling_agent_executor`
        # We can either extract it from llm object or just state the default.
        # For now, let's assume we know the default model or can add a way to get it.
        # default_model_name = agent_executor.agent.llm.model_name # This might work depending on LLM wrapper
        # Let's hardcode it for now based on what we set in function_calling_chain.py
        st.info(f"Using LLM Model (default for agent): `gpt-3.5-turbo-0125`") 

    except Exception as e:
        st.error(f"Failed to initialize Function-Calling Agent Executor: {e}")
        st.exception(e)
        return

    st.subheader("Query the Agent")
    
    # Initialize chat history in session state if not present
    if 'llm_chat_history' not in st.session_state:
        st.session_state.llm_chat_history = []

    # Display chat history
    for message in st.session_state.llm_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "intermediate_steps" in message and message["intermediate_steps"]:
                with st.expander("Show Agent's Work (Intermediate Steps)"):
                    st.json(message["intermediate_steps"])


    user_query = st.chat_input("Your query for the agent (e.g., 'What's the weather in Paris?', 'What is 15 * 24?'):")

    if user_query:
        st.session_state.llm_chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Agent is thinking..."):
            try:
                # Invoke the agent. The AgentExecutor should return intermediate_steps.
                # The create_openai_functions_agent is designed to work with AgentExecutor
                # and the executor should provide 'intermediate_steps' in its output dictionary.
                response = agent_executor.invoke({
                    "input": user_query,
                    # Passing actual message objects for chat history if agent supports it
                    # "chat_history": st.session_state.llm_chat_history # This can be complex to manage correctly
                                                                     # For now, let's keep it simple and stateless per query for display
                                                                     # or pass a simplified history if the agent expects it.
                                                                     # The current agent prompt has a placeholder but isn't strongly conversational.
                    "chat_history": [] # Keeping it simple for now, not truly conversational across multiple turns via this UI yet.
                })
                
                agent_final_answer = response.get("output", "No output found.")
                intermediate_steps = response.get("intermediate_steps", [])

                st.session_state.llm_chat_history.append({
                    "role": "assistant", 
                    "content": agent_final_answer,
                    "intermediate_steps": intermediate_steps
                })
                
                # Display assistant response (already done by iterating session_state now)
                # with st.chat_message("assistant"):
                #    st.markdown(agent_final_answer)
                #    if intermediate_steps:
                #        with st.expander("Show Agent's Work (Intermediate Steps)"):
                #            st.json(intermediate_steps) # st.json is good for structured data

                # Rerun to update the chat display immediately
                st.rerun()


            except Exception as e:
                st.error(f"Error during agent execution: {e}")
                st.exception(e)
                st.session_state.llm_chat_history.append({"role": "assistant", "content": f"Error: {e}", "intermediate_steps": []})
                st.rerun()


# --- Main app structure ---
st.set_page_config(layout="wide")

st.sidebar.title("Navigation")
page_options = {
    "Dashboard": main_dashboard,
    "Prompt Viewer": prompt_viewer_page,
    "Embedding Inspector": embedding_inspector_page,
    "LLM Interaction": llm_interaction_page,
}

selected_page = st.sidebar.radio("Go to", list(page_options.keys()))

# Display the selected page
page_function = page_options.get(selected_page)
if page_function:
    page_function()
else:
    st.error("Page not found.")

# To run this app:
# 1. Ensure you are in the `langchain_project` directory.
# 2. Run `streamlit run frontend/streamlit_app.py`
