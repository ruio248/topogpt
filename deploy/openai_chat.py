import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class OpenAIChatClient:
    """
    A class for managing OpenAI chat-based interactions.

    Attributes:
    -----------
    api_key : str
        The API key to authenticate requests.
    api_base : str
        The base URL of the API server.
    client : OpenAI
        An instance of the OpenAI client for making API calls.

    Methods:
    --------
    predict(message, history):
        Sends a user message and conversation history to the OpenAI API
        and returns the assistant's response.
    """

    def __init__(self):
        """
        Initializes the OpenAIChatClient with API key and base URL loaded from environment variables.
        """
        self.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        self.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")

        if self.api_key == "EMPTY":
            raise ValueError("API key is missing. Please set 'OPENAI_API_KEY' in your environment variables.")

        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def predict(self, message, history):
        """
        Sends a user message and conversation history to the OpenAI API.

        Parameters:
        -----------
        message : str
            The latest user message.
        history : list of tuples
            A list of previous interactions in the form [(user, assistant), ...].

        Returns:
        --------
        str
            The assistant's response to the user's message.
        """
        # Prepare conversation history in OpenAI format
        history_openai_format = [{"role": "system", "content": "You are a helpful assistant."}]
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})
        history_openai_format.append({"role": "user", "content": message})

        # Send API request
        response = self.client.chat.completions.create(
            model='your model name',  
            messages=history_openai_format,
            temperature=0.8,
            extra_body={
                'repetition_penalty': 1,
                'stop_token_ids': [7]
            }
        )
        # Extract the assistant's response
        return response.choices[0].message.content


if __name__ == "__main__":
    """
    Example usage and testing of the OpenAIChatClient class.
    """
    # Initialize the chat client
    chat_client = OpenAIChatClient()

    # Define the user's input message
    user_message = "Who do you think is the most outstanding football star in the world today?"

    # Example conversation history
    conversation_history = [
        ("Hello, who won the World Cup in 2018?", "France won the 2018 FIFA World Cup."),
        ("Who scored the most goals?", "Harry Kane was the top scorer in the 2018 World Cup.")
    ]

    # Generate a response
    response = chat_client.predict(user_message, conversation_history)
    print("Assistant's Response:")
    print(response)
