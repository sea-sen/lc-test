# Advanced Langchain Project Framework

This project provides a foundational framework for building complex applications using Langchain. It includes modules for prompt management, text embeddings, custom tool creation for function calling, and a core orchestrator to demonstrate their integration.

## Project Structure

```
langchain_project/
├── .env.example                # Example environment variables (copy to .env)
├── requirements.txt            # Project dependencies
├── core/                       # Core application logic and orchestration
│   └── app_controller.py       # Main demo script showcasing framework features
├── prompts/                    # Directory for storing prompt templates
│   ├── joke_prompt.json
│   └── summary_prompt.json
├── embeddings/                 # Embedding generation and management
│   └── embedding_manager.py
├── tools/                      # Custom tools for Langchain agents/chains
│   └── custom_tools.py
├── chains/                     # Langchain chains, including function calling agents
│   └── function_calling_chain.py
├── utils/                      # Utility scripts
│   └── prompt_loader.py
├── configs/                    # (Currently unused, .env is primary for now)
└── data/                       # (Placeholder for data files)
```

## Features

*   **Modular Structure:** Organized into components for easier management and scalability.
*   **Prompt Management:** Load prompt templates from external JSON files.
*   **Embedding Capabilities:** Generate text embeddings using OpenAI models and perform similarity searches with FAISS.
*   **Function Calling:** Demonstrates creating custom tools and using them with an OpenAI Functions Agent.
*   **Configuration Management:** Uses a `.env` file for API keys and other sensitive configurations.
*   **Core Orchestrator:** `core/app_controller.py` shows how to use the different components together.

## Setup

1.  **Clone the repository (or ensure you have the files as structured above).**

2.  **Create and Configure Environment File:**
    *   In the `langchain_project` directory, copy the `.env.example` file to a new file named `.env`.
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and replace `"your_openai_api_key_here"` with your actual OpenAI API key.
        ```
        OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```

3.  **Install Dependencies:**
    Navigate to the `langchain_project` directory in your terminal and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(It's highly recommended to use a Python virtual environment for this.)*

## How to Run

### Main Application Controller

The primary way to see the framework in action is by running the `app_controller.py` script. This script demonstrates several features interactively.

1.  Make sure you have completed the Setup steps (especially `.env` file and dependencies).
2.  Navigate to the `langchain_project` directory.
3.  Run the application controller:
    ```bash
    python core/app_controller.py
    ```
    The script will guide you through:
    *   Generating a joke based on a topic you provide (uses prompt loading).
    *   Performing a similarity search on a small dataset using embeddings.
    *   Interacting with the function-calling agent.

### Individual Component Tests/Examples

Some modules contain `if __name__ == '__main__':` blocks that allow you to test their specific functionality directly. This can be useful for development and debugging.

*   **Prompt Loader:**
    ```bash
    python utils/prompt_loader.py 
    ```
    (Ensure your current directory is `langchain_project` or adjust paths in the script if running from `utils/` directly and it has issues finding `prompts/`)

*   **Embedding Manager:**
    ```bash
    python embeddings/embedding_manager.py
    ```
    (Requires `OPENAI_API_KEY` in `.env`)

*   **Custom Tools:**
    ```bash
    python tools/custom_tools.py
    ```

*   **Function Calling Chain:**
    ```bash
    python chains/function_calling_chain.py
    ```
    (Requires `OPENAI_API_KEY` in `.env`)

**Note on Running Individual Scripts:** When running scripts from subdirectories directly, Python's module resolution might sometimes require you to set the `PYTHONPATH` to include the project's root directory (`langchain_project`) or run them as modules (e.g., `python -m utils.prompt_loader`). The `app_controller.py` script includes logic to handle `sys.path` correctly, making it the most robust way to run the integrated demo.

## Key Components Explained

*   **`prompts/` & `utils/prompt_loader.py`**:
    *   Stores prompt templates in structured JSON files.
    *   `prompt_loader.py` provides `load_prompt_template` to easily load these into Langchain `PromptTemplate` objects.

*   **`embeddings/embedding_manager.py`**:
    *   `EmbeddingManager` class handles initialization of embedding models (e.g., `OpenAIEmbeddings`).
    *   Provides methods to get embeddings for texts.
    *   The `if __name__ == '__main__':` block shows how to use it with FAISS for similarity search.

*   **`tools/custom_tools.py`**:
    *   Defines custom tools (`simple_calculator`, `get_mock_weather`) using Langchain's `@tool` decorator and Pydantic models for input schema. These tools can be used by Langchain agents.

*   **`chains/function_calling_chain.py`**:
    *   Demonstrates how to create an agent that can use the custom tools. It uses `create_openai_functions_agent` and `AgentExecutor`.

*   **`core/app_controller.py`**:
    *   The central script that ties together various components to showcase a sample application flow. It's the best place to start understanding how the pieces connect.

*   **`.env` / `python-dotenv`**:
    *   Used for managing environment variables, primarily your `OPENAI_API_KEY`. `load_dotenv()` is called in relevant scripts to load these variables.

## Streamlit Frontend UI

This project includes an interactive web interface built with Streamlit to help you view configurations, interact with components, and audit behavior.

### Running the Streamlit App

1.  **Ensure Setup is Complete:** Make sure you have completed all steps in the main "Setup" section of this README (Python environment, `.env` file with `OPENAI_API_KEY`, and `pip install -r requirements.txt`). `streamlit` is included in `requirements.txt`.

2.  **Navigate to Project Root:** Open your terminal and change to the `langchain_project` directory.

3.  **Run Streamlit:** Execute the following command:
    ```bash
    streamlit run frontend/streamlit_app.py
    ```
    This will typically open the application in your default web browser.

### UI Features

The Streamlit application ("Langchain Project Control Center") provides the following sections (accessible via the sidebar):

*   **Dashboard:** An overview page showing API key status and brief descriptions of other sections.
*   **Prompt Viewer:**
    *   Lists all available prompt templates (from `prompts/*.json`).
    *   Allows you to select a prompt and view its details: input variables, template string, and raw JSON source.
*   **Embedding Inspector:**
    *   Displays the configured OpenAI embedding model.
    *   Allows you to input text and generate its embedding, showing dimensionality and a snippet of the vector.
    *   Includes a simple similarity search demo against a predefined set of documents using FAISS.
*   **LLM Interaction:**
    *   Provides a chat interface to interact with the function-calling agent.
    *   Displays the agent's final answers.
    *   Shows the intermediate steps (tool calls and agent observations) for transparency, helping you audit the agent's reasoning process.

This UI is designed to make the underlying Langchain components more accessible and easier to understand.

This framework is a starting point. You can expand it by adding more prompts, tools, chains, data sources, and more sophisticated orchestration logic.
