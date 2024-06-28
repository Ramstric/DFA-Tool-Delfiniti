import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import sv_ttk

import numpy as np
import torch

from ctypes import windll

# Set the DPI awareness (high resolution for fonts) for the current process
windll.shcore.SetProcessDpiAwareness(1)


def import_file():
    file_path = filedialog.askopenfilename(title="Select a file",
                                           filetypes=[("Text files", "*.txt")])
    if file_path:
        # Process the selected file (you can replace this with your own logic)
        print("Selected file:", file_path)
        txt_file = open(file_path, "r")
        txt_data = np.loadtxt(txt_file).astype(np.float32)
        np_x, np_t = np.split(txt_data, 2, axis=1)
        np_x = np_x.flatten()
        np_t = np_t.flatten()

        x = torch.from_numpy(np_x)
        t = torch.from_numpy(np_t)
        print(x)


# Create the main Tkinter window
root = tk.Tk()
root.option_add("*Font", "Arial")

root.title("DFA Tool")
root.geometry("400x400")

# Create an "Import File" button
import_button = ttk.Button(root, text="Import File .txt", command=import_file)
import_button.pack(pady=20)

sv_ttk.set_theme("dark")

# Run the Tkinter event loop
root.mainloop()
