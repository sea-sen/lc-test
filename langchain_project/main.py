import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    # Remind the user to set their API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this application.")
        print("You can get an API key from https://platform.openai.com/account/api-keys")
        return

    # Initialize the OpenAI LLM
    llm = OpenAI(openai_api_key=api_key, temperature=0.7)

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template="Tell me a short joke about {topic}."
    )

    # Create an LLMChain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Get user input
    try:
        topic_input = input("Enter a topic for a joke (e.g., 'cats', 'programmers'): ")
        if not topic_input:
            print("No topic entered. Exiting.")
            return
    except EOFError:
        print("
No input received. Exiting.")
        return


    # Run the chain and print the response
    try:
        response = chain.invoke({"topic": topic_input})
        if isinstance(response, dict) and 'text' in response:
            print("\nJoke:")
            print(response['text'])
        else:
            print("\nUnexpected response format:")
            print(response)
    except Exception as e:
        print(f"An error occurred while generating the joke: {e}")

if __name__ == "__main__":
    main()
