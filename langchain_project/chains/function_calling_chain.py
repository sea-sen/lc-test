import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage # Keep if used in __main__
from dotenv import load_dotenv

# Load .env file from project root.
# This assumes the module is part of `langchain_project`.
PROJECT_ROOT_CHAINS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path_chains = os.path.join(PROJECT_ROOT_CHAINS, '.env')
load_dotenv(dotenv_path=dotenv_path_chains)


# Assuming custom_tools.py is in langchain_project/tools/
# Adjust path if necessary based on execution context
from ..tools.custom_tools import simple_calculator, get_mock_weather


def create_function_calling_agent_executor(api_key: str = None, model_name: str = "gpt-3.5-turbo-0125"):
    """
    Creates an OpenAI Functions Agent Executor with the simple_calculator and get_mock_weather tools.
    Tries to load OPENAI_API_KEY from .env file or environment variables if api_key is not provided.
    """
    # load_dotenv() # Called globally at module load
    
    actual_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
    if not actual_api_key:
        raise ValueError("OPENAI_API_KEY not provided directly or found in .env file or environment variables for ChatOpenAI.")

    llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=actual_api_key)
    tools = [simple_calculator, get_mock_weather]

    # This prompt is a basic structure. You might want to customize it further.
    # It's important to include a placeholder for agent_scratchpad for the agent to work.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools. Do your best to answer the user's questions."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

if __name__ == '__main__':
    print("Testing OpenAI Functions Agent with Custom Tools...")
    
    # IMPORTANT: Ensure OPENAI_API_KEY is set in your environment variables or .env file
    try:
        # load_dotenv() is called at module level
        agent_executor = create_function_calling_agent_executor()
        print("Agent Executor created.")
    except ValueError as e:
        print(f"Error initializing agent executor: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable or in your .env file.")
        print("Skipping further demo.")
        exit()
    
    chat_history = []

    queries = [
        "What's the weather like in Paris?",
        "Calculate 25 multiplied by 4, and then add 15 to the multiplication result.", # More complex, requires thought or multiple steps
        "What is 100 divided by 4?",
        "What about Tokyo's weather?",
        "Add 50 and 30, then tell me the weather in the city that is the capital of Japan based on that sum (just kidding, just tell me the weather in Tokyo).", # Test robustness
    ]

    for query_text in queries:
        print(f"\n--- User Query: {query_text} ---")
        try:
            # For more complex interactions, you'd manage chat_history properly.
            # Here, we'll pass it but it won't be strongly utilized by this specific agent prompt for history.
            # A more conversational agent would require a prompt designed for chat history.
            response = agent_executor.invoke({
                "input": query_text,
                "chat_history": chat_history 
            })
            print(f"\nAgent Response: {response.get('output')}")
            
            # Simple way to add to history for this demo (won't be used by agent effectively without prompt changes)
            # chat_history.append(HumanMessage(content=query_text))
            # chat_history.append(AIMessage(content=str(response.get('output'))))

        except Exception as e:
            print(f"An error occurred while processing query '{query_text}': {e}")
