"""
A modern GUI chatbot with vision capabilities using Ollama's vision model.
Features:
- Drag and drop image support
- Chat history with scrollable view
- Modern dark theme UI
- Async message handling
- Image preview
"""

import asyncio
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import ollama
from typing import Optional, List
from dataclasses import dataclass
import os
from datetime import datetime
import base64
import nest_asyncio
import threading
from queue import Queue

# Enable nested event loops
nest_asyncio.apply()

@dataclass
class Message:
    """Represents a chat message."""
    role: str
    content: str
    timestamp: str
    image_path: Optional[str] = None

class VisionChatGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Vision Chat Assistant")
        self.window.geometry("1000x800")
        ctk.set_appearance_mode("dark")
        
        self.messages: List[Message] = []
        self.client = ollama.AsyncClient()
        self.current_image: Optional[str] = None
        self.message_queue = Queue()
        
        # Create event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        self.setup_gui()
        
    def _run_event_loop(self):
        """Run asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def setup_gui(self):
        """Setup the GUI components."""
        # Main container
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat history frame
        self.chat_frame = ctk.CTkScrollableFrame(
            self.main_container,
            height=500
        )
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image preview frame
        self.preview_frame = ctk.CTkFrame(self.main_container)
        self.preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.image_label = ctk.CTkLabel(self.preview_frame, text="No image selected")
        self.image_label.pack(pady=5)
        
        # Input frame
        input_frame = ctk.CTkFrame(self.main_container)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Message input
        self.message_input = ctk.CTkTextbox(input_frame, height=60)
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Buttons frame
        button_frame = ctk.CTkFrame(input_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # Image button
        self.image_button = ctk.CTkButton(
            button_frame,
            text="Add Image",
            command=self.select_image
        )
        self.image_button.pack(pady=(0, 5))
        
        # Send button
        self.send_button = ctk.CTkButton(
            button_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack()
        
        # Bind enter key to send message
        self.window.bind('<Return>', lambda e: self.send_message())
        
        # Start message processing
        self.window.after(100, self.process_message_queue)
        
    def select_image(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")
            ]
        )
        if file_path:
            self.current_image = file_path
            self.update_image_preview(file_path)
            
    def update_image_preview(self, image_path: str):
        """Update the image preview."""
        try:
            image = Image.open(image_path).convert('RGB')
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            
            if hasattr(self, 'preview_image_label'):
                self.preview_image_label.destroy()
                
            self.preview_image_label = tk.Label(
                self.preview_frame,
                image=photo,
                bg='#2b2b2b'
            )
            self.preview_image_label.image = photo
            self.preview_image_label.pack(pady=5)
            self.image_label.configure(text=os.path.basename(image_path))
            
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {str(e)}")
            
    def add_message_to_chat(self, message: Message):
        """Add a message to the chat history."""
        msg_frame = ctk.CTkFrame(
            self.chat_frame,
            fg_color='#2b2b2b'
        )
        msg_frame.pack(fill=tk.X, padx=5, pady=2)
        
        header = ctk.CTkLabel(
            msg_frame,
            text=f"{message.timestamp} - {message.role}:",
            font=("Arial", 10),
            fg_color='transparent'
        )
        header.pack(anchor="w")
        
        content = ctk.CTkLabel(
            msg_frame,
            text=message.content,
            wraplength=600,
            justify="left",
            fg_color='transparent'
        )
        content.pack(anchor="w", padx=10)
        
        if message.image_path:
            try:
                image = Image.open(message.image_path).convert('RGB')
                image.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(image)
                
                image_label = tk.Label(
                    msg_frame,
                    image=photo,
                    bg='#2b2b2b'
                )
                image_label.image = photo
                image_label.pack(padx=10, pady=5)
            except Exception as e:
                print(f"Error displaying message image: {str(e)}")
                
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    async def get_ai_response(self, user_message: str, image_path: Optional[str] = None):
        """Get response from AI model."""
        try:
            messages = [{"role": "user", "content": user_message}]
            
            if image_path:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode()
                    messages[0]["images"] = [image_data]
            
            response = await self.client.chat(
                model="llama3.2-vision",
                messages=messages
            )
            
            return response.message.content
        except Exception as e:
            return f"Error: {str(e)}"
            
    def process_message_queue(self):
        """Process messages in the queue."""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self.add_message_to_chat(message)
        finally:
            self.window.after(100, self.process_message_queue)
            
    def send_message(self):
        """Send message and get AI response."""
        message_text = self.message_input.get("1.0", tk.END).strip()
        if not message_text and not self.current_image:
            return
            
        # Clear input
        self.message_input.delete("1.0", tk.END)
        
        # Add user message to chat
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_message = Message(
            role="User",
            content=message_text,
            timestamp=timestamp,
            image_path=self.current_image
        )
        self.message_queue.put(user_message)
        
        # Disable input while processing
        self.message_input.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.image_button.configure(state="disabled")
        
        # Create async task
        async def process_response():
            response_text = await self.get_ai_response(message_text, self.current_image)
            
            # Add AI response to chat
            ai_message = Message(
                role="Assistant",
                content=response_text,
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
            self.message_queue.put(ai_message)
            
            # Re-enable input in main thread
            self.window.after(0, lambda: [
                self.message_input.configure(state="normal"),
                self.send_button.configure(state="normal"),
                self.image_button.configure(state="normal")
            ])
            
            # Clear current image
            self.current_image = None
            if hasattr(self, 'preview_image_label'):
                self.window.after(0, self.preview_image_label.destroy)
            self.window.after(0, lambda: self.image_label.configure(text="No image selected"))
            
        # Schedule the coroutine in the event loop
        asyncio.run_coroutine_threadsafe(process_response(), self.loop)
        
    def run(self):
        """Start the GUI application."""
        self.window.mainloop()
        
    def cleanup(self):
        """Cleanup resources."""
        self.loop.stop()
        self.thread.join(timeout=1)

if __name__ == "__main__":
    app = VisionChatGUI()
    try:
        app.run()
    finally:
        app.cleanup()
