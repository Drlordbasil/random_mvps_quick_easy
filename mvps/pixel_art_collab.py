import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from typing import Optional, List, Dict, Any
from groq import Groq
import base64
from io import BytesIO
import json

class AIFeedbackPanel(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.setup_ui()
        self.processing = False
        self.current_suggestion = None

    def setup_ui(self):
        self.feedback_label = ctk.CTkLabel(self, text="AI Feedback", font=("Arial", 16, "bold"))
        self.feedback_label.pack(pady=5)

        self.feedback_text = ctk.CTkTextbox(self, width=300, height=150)
        self.feedback_text.pack(pady=5, padx=10)

        self.apply_button = ctk.CTkButton(self, text="Apply Suggestion", command=self.apply_suggestion)
        self.apply_button.pack(pady=5)
        self.apply_button.configure(state="disabled")

        self.clear_button = ctk.CTkButton(self, text="Clear Suggestion", command=self.clear_suggestion)
        self.clear_button.pack(pady=5)
        self.clear_button.configure(state="disabled")

    def set_processing(self, is_processing: bool):
        self.processing = is_processing
        state = "disabled" if is_processing else "normal"
        self.apply_button.configure(state=state)
        self.clear_button.configure(state=state)

    def display_feedback(self, feedback: str, suggestion: Dict[str, Any]):
        self.current_suggestion = suggestion
        self.feedback_text.delete("1.0", tk.END)
        
        # Format and display the feedback in a structured way
        feedback_text = f"""Overall Assessment:
{feedback}

Suggested Action:
{suggestion.get('description', 'No specific action suggested')}

Action Type: {suggestion.get('action_type', 'none')}
Color: {suggestion.get('color', 'N/A')}
Affected Areas: {len(suggestion.get('coordinates', []))} points
"""
        self.feedback_text.insert("1.0", feedback_text)
        
        # Enable buttons if we have a valid suggestion
        if suggestion and suggestion.get('action_type') != 'none':
            self.apply_button.configure(state="normal")
            self.clear_button.configure(state="normal")
        else:
            self.apply_button.configure(state="disabled")
            self.clear_button.configure(state="disabled")

    def apply_suggestion(self):
        if self.current_suggestion:
            self.app.apply_ai_suggestion(self.current_suggestion)

    def clear_suggestion(self):
        self.current_suggestion = None
        self.feedback_text.delete("1.0", tk.END)
        self.apply_button.configure(state="disabled")
        self.clear_button.configure(state="disabled")

class PixelCanvas(tk.Canvas):
    def __init__(self, master, width=400, height=400, pixel_size=20):
        super().__init__(master, width=width, height=height, bg="white", highlightthickness=1)
        self.pixel_size = pixel_size
        self.width = width
        self.height = height
        self.rows = height // pixel_size
        self.cols = width // pixel_size
        self.current_color = "#000000"
        self.pixels = np.full((self.rows, self.cols, 3), 255, dtype=np.uint8)
        self.setup_canvas()
        self.bind_events()

    def setup_canvas(self):
        self.draw_grid()

    def draw_grid(self):
        for i in range(0, self.width, self.pixel_size):
            self.create_line(i, 0, i, self.height, fill="gray")
        for i in range(0, self.height, self.pixel_size):
            self.create_line(0, i, self.width, i, fill="gray")

    def bind_events(self):
        self.bind("<B1-Motion>", self.paint_pixel)
        self.bind("<Button-1>", self.paint_pixel)

    def paint_pixel(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.set_pixel(row, col, self.current_color)

    def set_color(self, color):
        self.current_color = color

    def get_image(self):
        img = Image.fromarray(self.pixels.astype('uint8'))
        return img

    def clear(self):
        self.pixels.fill(255)
        self.delete("all")
        self.draw_grid()

    def set_pixel(self, row, col, color):
        if not isinstance(color, str) or not color.startswith('#'):
            return
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            self.pixels[row, col] = [r, g, b]
            x1 = col * self.pixel_size
            y1 = row * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        except (ValueError, IndexError):
            pass

class ColorPalette(ctk.CTkFrame):
    def __init__(self, master, canvas):
        super().__init__(master)
        self.canvas = canvas
        self.colors = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFFFFF"]
        self.setup_palette()

    def setup_palette(self):
        for i, color in enumerate(self.colors):
            btn = ctk.CTkButton(
                self,
                text="",
                width=30,
                height=30,
                fg_color=color,
                command=lambda c=color: self.canvas.set_color(c)
            )
            btn.grid(row=0, column=i, padx=2, pady=2)

class ArtAnalyzer:
    def __init__(self, client: Groq):
        self.client = client
        self.model = 'llama-3.2-11b-vision-preview'

    def encode_image(self, img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_image(self, img: Image.Image) -> Dict[str, Any]:
        base64_image = self.encode_image(img)
        user_prompt = '''Analyze this pixel art and provide specific suggestions for improvements. 
        Format your response as a structured analysis with the following sections:
        1. Overall Assessment
        2. Specific Suggestions
        3. Action Plan
        
        Example Response Structure:
        {
            "feedback": {
                "overall_assessment": "The pixel art shows good use of basic shapes but could benefit from more defined edges.",
                "color_palette": "Limited color range, consider adding more contrast",
                "composition": "Well-centered but lacks depth"
            },
            "suggestion": {
                "action_type": "draw",  // Can be: "draw", "clear", "change_color"
                "description": "Add shading to the main figure",
                "coordinates": [
                    {"row": 5, "col": 5, "purpose": "shadow edge"},
                    {"row": 6, "col": 5, "purpose": "shadow fill"}
                ],
                "color": "#2A2A2A"
            }
        }'''
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                model=self.model,
                temperature=0.7
            )
            
            response_text = chat_completion.choices[0].message.content
            print(f"Raw response: {response_text}")  # Debug print
            
            try:
                # Try to parse the response as JSON first
                response_data = json.loads(response_text)
                return response_data
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response from the text
                return {
                    "feedback": {
                        "overall_assessment": response_text,
                        "color_palette": "Analysis not structured",
                        "composition": "Analysis not structured"
                    },
                    "suggestion": {
                        "action_type": "draw",
                        "description": "AI suggestion needs review",
                        "coordinates": [{"row": 5, "col": 5, "purpose": "suggestion point"}],
                        "color": "#FF0000"
                    }
                }
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                "feedback": {
                    "overall_assessment": "Error analyzing image",
                    "color_palette": "Error",
                    "composition": "Error"
                },
                "suggestion": {
                    "action_type": "none",
                    "description": "Error occurred during analysis",
                    "coordinates": [],
                    "color": None
                }
            }

class ArtApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Co-Artist: Collaborative Pixel Art Studio")
        self.client = Groq()
        self.art_analyzer = ArtAnalyzer(self.client)
        self.setup_ui()

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=10, pady=10, expand=True, fill="both")

        canvas_frame = ctk.CTkFrame(main_frame)
        canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = PixelCanvas(canvas_frame)
        self.canvas.pack()

        self.color_palette = ColorPalette(canvas_frame, self.canvas)
        self.color_palette.pack(pady=10)

        button_frame = ctk.CTkFrame(canvas_frame)
        button_frame.pack(pady=10)

        ctk.CTkButton(button_frame, text="Clear", command=self.canvas.clear).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Save", command=self.save_artwork).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Get AI Feedback", command=self.get_ai_feedback).pack(side=tk.LEFT, padx=5)

        self.feedback_panel = AIFeedbackPanel(main_frame, self)
        self.feedback_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill="y")

    def save_artwork(self):
        img = self.canvas.get_image()
        img.save("artwork.png")

    def get_ai_feedback(self):
        self.feedback_panel.set_processing(True)
        img = self.canvas.get_image()
        response = self.art_analyzer.analyze_image(img)
        if response:
            self.feedback_panel.display_feedback(
                response.get("feedback", {}).get("overall_assessment", "No feedback available"),
                response.get("suggestion", {})
            )
        else:
            self.feedback_panel.display_feedback("Failed to get AI analysis", {})
        self.feedback_panel.set_processing(False)

    def apply_ai_suggestion(self, suggestion: Dict[str, Any]):
        action = suggestion.get("action_type")
        if not action:
            return

        if action == "clear":
            self.canvas.clear()
        elif action == "draw":
            color = suggestion.get("color")
            coordinates = suggestion.get("coordinates", [])
            if color and coordinates:
                for coord in coordinates:
                    row = coord.get("row")
                    col = coord.get("col")
                    if row is not None and col is not None:
                        self.canvas.set_pixel(row, col, color)
        elif action == "change_color":
            color = suggestion.get("color")
            if color:
                self.canvas.set_color(color)

if __name__ == "__main__":
    app = ArtApp()
    app.mainloop()
