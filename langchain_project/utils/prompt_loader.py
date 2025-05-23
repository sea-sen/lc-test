import json
from langchain.prompts import PromptTemplate
import os

def load_prompt_template(prompt_name: str, prompts_dir: str = "prompts") -> PromptTemplate:
    """
    Loads a prompt template from a JSON file in the specified directory.

    Args:
        prompt_name (str): The name of the prompt (without .json extension).
        prompts_dir (str): The directory where prompt files are stored. 
                           Assumed to be relative to the project root or an absolute path.

    Returns:
        PromptTemplate: A Langchain PromptTemplate object.

    Raises:
        FileNotFoundError: If the prompt file is not found.
        KeyError: If the JSON file doesn't contain expected keys.
    """
    # Construct the full path to the prompt file
    # This assumes the prompts_dir is accessible from where this script is run
    # For robustness in a larger project, consider using absolute paths
    # based on the project's root directory.
    
    # Assuming this util might be called from core or other places,
    # let's try to make prompts_dir relative to the project root if it's not absolute.
    # This is a simplification; a more robust solution might use a global config for base_path.
    
    # Simplified path construction:
    # For this subtask, we'll assume prompts_dir is relative to `langchain_project`
    # and the calling script is also run from `langchain_project` or `langchain_project/core`.
    # A more robust path handling would be needed for a real application.
    
    # Let's find the project root based on a known file/dir, e.g. `requirements.txt`
    # This is a bit of a heuristic.
    current_path = os.path.abspath(os.path.dirname(__file__)) # utils directory
    project_root = os.path.abspath(os.path.join(current_path, '..')) # langchain_project directory
    
    file_path = os.path.join(project_root, prompts_dir, f"{prompt_name}.json")

    if not os.path.exists(file_path):
         # Fallback for cases where script is run from langchain_project root
        alt_file_path = os.path.join(prompts_dir, f"{prompt_name}.json")
        if os.path.exists(alt_file_path):
            file_path = alt_file_path
        else:
            raise FileNotFoundError(f"Prompt file {prompt_name}.json not found at {file_path} or {alt_file_path}")

    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the prompt template file: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from the prompt template file: {file_path}")

    input_vars = config.get("input_variables", [])
    template_str = config.get("template", "")

    if not template_str:
        raise ValueError(f"Template string is missing in {file_path}")

    return PromptTemplate(input_variables=input_vars, template=template_str)

if __name__ == '__main__':
    # Example Usage (assuming you run this from `langchain_project/utils/` or `langchain_project/`)
    # To make this runnable from utils, adjust path for prompts_dir
    try:
        # Test when running from `langchain_project/utils/`
        print("Attempting to load 'joke_prompt'...")
        # The load_prompt_template function calculates project_root internally.
        # The default prompts_dir="prompts" should work correctly when this script
        # is located in langchain_project/utils/
        joke_tpl = load_prompt_template("joke_prompt") 
        print("Joke Prompt loaded successfully:")
        print(f"  Input Variables: {joke_tpl.input_variables}")
        print(f"  Template: {joke_tpl.template}")
        print(joke_tpl.format(topic="cats"))

        print("\nAttempting to load 'summary_prompt'...")
        summary_tpl = load_prompt_template("summary_prompt")
        print("Summary Prompt loaded successfully:")
        print(f"  Input Variables: {summary_tpl.input_variables}")
        print(f"  Template: {summary_tpl.template}")
        print(summary_tpl.format(text_to_summarize="This is a long text..."))

    except Exception as e:
        print(f"Error during example usage: {e}")
        print("Note: If running this script directly, ensure your CWD or paths are set up correctly.")
        print(f"Current working directory: {os.getcwd()}")
        # For printing project_root in this context, we recalculate it as it's scoped within the function.
        current_path_for_main = os.path.abspath(os.path.dirname(__file__))
        project_root_for_main = os.path.abspath(os.path.join(current_path_for_main, '..'))
        print(f"Calculated project_root (for this __main__ block context): {project_root_for_main}")
