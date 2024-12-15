"""
A delightful chatbot that maintains conversation history and aims to please the user.
Built using Ollama's powerful language models.

Features:
- Maintains chat history for contextual conversations
- Graceful error handling and exit
- Rich conversation formatting
- Customizable system prompt
- Async support for better performance
"""

import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from rich.panel import Panel
import colorama

# Initialize colorama for Windows console
colorama.init()

@dataclass
class Message:
    """Represents a chat message with role and content."""
    role: str
    content: str

class ChatBot:
    """A friendly chatbot that maintains conversation history."""
    
    def __init__(self, model: str = "llama3.2"):
        """Initialize the chatbot with specified model."""
        self.model = model
        self.client = ollama.AsyncClient()
        self.console = Console()
        self.messages: List[Message] = []
        self.system_prompt = """You are a delightful and enthusiastic AI assistant. 
        Your responses should be helpful, engaging, and aim to make the user happy. 
        Feel free to use friendly language, but maintain professionalism."""
        
    async def initialize(self) -> None:
        """Initialize the chat session with system prompt."""
        self.messages.append(Message(role="system", content=self.system_prompt))
        
    def format_response(self, content: str) -> None:
        """Format and display the AI response."""
        try:
            self.console.print(Panel(
                Markdown(content),
                title="AI Assistant",
                border_style="blue"
            ))
        except Exception as e:
            # Fallback to simple printing if rich formatting fails
            print("\nAI Assistant:", content, "\n")

    async def get_response(self, user_input: str) -> Optional[str]:
        """Get response from the AI model."""
        try:
            # Add user message to history
            self.messages.append(Message(role="user", content=user_input))
            
            # Convert messages to format expected by Ollama
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages
            ]
            
            # Get AI response
            response = await self.client.chat(
                model=self.model,
                messages=formatted_messages
            )
            
            # Add AI response to history
            content = response.message.content
            self.messages.append(Message(role="assistant", content=content))
            return content
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

async def main():
    """Main chat loop."""
    chatbot = ChatBot()
    await chatbot.initialize()
    
    # Welcome message
    print("\nWelcome to the Friendly Chat Bot!")
    print("Type 'exit' or press Ctrl+C to end the chat.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Have a wonderful day!")
                break
                
            # Get and display AI response
            response = await chatbot.get_response(user_input)
            if response:
                chatbot.format_response(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye! Have a wonderful day!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
