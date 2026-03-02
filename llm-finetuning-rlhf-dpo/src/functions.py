from transformers import pipeline

def template(idea):
    return f"""Given the YouTube video idea write an engaging title for it.

**Video Idea**: {idea.lower()}

**Additional Guidance**:
- Title should be between 30 and 75 characters long
- Only return the title idea, nothing else!
"""

def format_chat_prompt(user_input, system_message="You are a helpful assistent"):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user
    
    Returns:
        str: Formatted prompt for the model.
    """

    # Format user message
    user_prompt = f"<|im_start|>\n{user_input}<|im_end|>\n"

    # Start assistant's turn
    assistant_prompt = "<im_start>assistant\n"

    # Combine prompts
    formatted_prompt = user_input + assistant_prompt

    return formatted_prompt

def generate_title(idea, model, tokenizer, temperature=0.7, num_titles=1):
    """
    Generates a YouTube video title based on a given idea.

    Args:
        idea (str): The video idea to generate a title for
        model: The language model to use
        tokenizer: The tokenizer for the model
        max_length (int): The Maximum length of generated text
        temperature (float): Temperature for text generation
        num_titles (int): Number of titles to generate
    
    Returns:
        list: List of generated video titles
    """

    # Set up text generation pipline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # 