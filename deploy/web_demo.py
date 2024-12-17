import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChatbotInterface:
    """
    A class to encapsulate a Gradio chatbot interface using OpenAI's API, 
    specifically designed to interact with vLLM-deployed models.
    """

    def __init__(self):
        """
        Initializes the ChatbotInterface with OpenAI API configurations.
        Environment variables:
        - OPENAI_API_KEY: The API key for OpenAI.
        - OPENAI_API_BASE: The API base URL, typically the vLLM server endpoint.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE")
        if not self.api_key or not self.base_url:
            raise ValueError("Both OPENAI_API_KEY and OPENAI_API_BASE must be set in the environment.")

        # Initialize OpenAI client with API key and base URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def format_history(self, message, history):
        """
        Formats the chat history for OpenAI-compatible structure.

        Args:
            message (str): The latest user input message.
            history (list): A list of tuples [(user_message, assistant_response), ...]

        Returns:
            list: Formatted chat history including the current message.
        """
        formatted_history = [{"role": "system", "content": "You are a helpful and reliable AI assistant."}]
        for user_input, assistant_response in history:
            formatted_history.append({"role": "user", "content": user_input})
            formatted_history.append({"role": "assistant", "content": assistant_response})
        formatted_history.append({"role": "user", "content": message})
        return formatted_history

    def predict(self, message, history):
        """
        Handles the message prediction using the OpenAI API with streaming support.

        Args:
            message (str): The latest user input message.
            history (list): A list of previous chat history tuples.

        Yields:
            str: Partial generated messages in a streaming manner.
        """
        formatted_history = self.format_history(message, history)
        stream = self.client.chat.completions.create(
            model="Qwen1.5-7B-Chat",  # Model name as deployed on vLLM
            messages=formatted_history,
            temperature=0.8,
            stream=True,
            extra_body={
                "repetition_penalty": 1.0,
                "stop_token_ids": [7],
            }
        )

        partial_message = ""
        for chunk in stream:
            partial_message += chunk.choices[0].delta.content or ""
            yield partial_message

    def launch_interface(self):
        """
        Launches the Gradio chat interface.
        """
        print("Launching Gradio Chat Interface...")
        gr.ChatInterface(self.predict).queue().launch(share=True)


if __name__ == "__main__":
    # Initialize and launch the chatbot interface
    chatbot = ChatbotInterface()
    chatbot.launch_interface()
