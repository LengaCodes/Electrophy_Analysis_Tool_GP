import os 
import tkinter as tk
from tkinter import filedialog, messagebox
from .base_window import BaseWindow
from .visu_window import VisuWindow

class WelcomeWindow(BaseWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Title Label
        title_label = tk.Label(self.inner_frame, text="WELCOME", font=("Arial", 18), bg="#FFFFFF")
        title_label.pack(pady=20)

        # File selection label
        select_label = tk.Label(self.inner_frame, text="Select a file to analyze", font=("Arial", 12), bg="#FFFFFF")
        select_label.pack()

        # File selection button
        select_button = tk.Button(self.inner_frame, text="Select File", command=self.select_file, font=("Arial", 12))
        select_button.pack(pady=10)

        # Display filename
        self.file_label = tk.Label(self.inner_frame, text="", font=("Arial", 12), bg="#FFFFFF")
        self.file_label.pack(pady=10)

        # Next button
        self.next_button = tk.Button(self.inner_frame, text="NEXT", command=self.next_button_click, font=("Arial", 12), state=tk.DISABLED)
        self.next_button.pack(pady=20)

        self.selected_file_path = None

    def select_file(self):
        # Open file dialog and get the selected file
        file_path = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")])
        if file_path:
            # Extract only the file name from the full path
            file_name = os.path.basename(file_path)
            self.file_label.config(text=file_name)
            self.selected_file_path = file_path

            self.next_button.config(state=tk.NORMAL)

    def next_button_click(self):
        # Open VisuWindow if a file is selected
        if self.selected_file_path:
            self.destroy()
            # Pass the icon and image paths explicitly if needed
            app = VisuWindow(file_path=self.selected_file_path)
            app.mainloop()
        else:
            messagebox.showwarning("Warning", "Please select a file before proceeding.")
