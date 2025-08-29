import os
import sys
import tkinter as tk
from PIL import Image, ImageTk

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class BaseWindow(tk.Tk):
    def __init__(self, title="ELECTROPHYSIOLOGY SIGNAL ANALYSIS",
                 icon_path="media/logoClair.ico",
                 image_path="media/logoVclair.png",
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.title(title)
        self.geometry('600x400+50+50')
        self.resizable(False, False)

        self.iconbitmap(resource_path(icon_path))

        self.outer_frame = tk.Frame(self, bg='darkblue', bd=0)
        self.outer_frame.pack(fill='both', expand=True)

        self.inner_frame = tk.Frame(self.outer_frame, bg='white')
        self.inner_frame.pack(fill='both', expand=True, padx=5, pady=(5, 25))

        self.bottom_band = tk.Frame(self.inner_frame, bg='darkblue', height=20)
        self.bottom_band.pack(side='bottom', fill='x')

        self.left_label = tk.Label(self.bottom_band, text="Copyright - 2025", bg='darkblue', fg='white', font=('Arial', 9))
        self.left_label.pack(side='left', padx=10)

        self.set_image(image_path)

    def set_image(self, image_path):
        full_path = resource_path(image_path)
        if os.path.exists(full_path):
            original_image = Image.open(full_path)

            max_width, max_height = 150, 20
            ratio = min(max_width / original_image.width, max_height / original_image.height)
            resized_image = original_image.resize((int(original_image.width * ratio), int(original_image.height * ratio)), Image.LANCZOS)

            self.photo = ImageTk.PhotoImage(resized_image)
            image_label = tk.Label(self.bottom_band, image=self.photo, bg='darkblue')
            image_label.image = self.photo
            image_label.pack(side='right', padx=10)
