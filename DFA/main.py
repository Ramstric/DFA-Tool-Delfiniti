from tkinter import Tk, filedialog, IntVar
from tkinter import ttk
from threading import Thread
import sv_ttk

import numpy as np
import torch

from ctypes import windll

import DFA

# Set the DPI awareness (high resolution for fonts) for the current process
windll.shcore.SetProcessDpiAwareness(1)


class App:
    def __init__(self, master: Tk) -> None:
        self.file_name = None
        self.txt_path = None
        self.save_path = None
        self.x, self.y = None, None
        self.master = master

        # Create an "Import File" button
        self.import_button = ttk.Button(root, text="Import File .txt", command=self.import_file_data)
        self.import_button.grid(row=1, column=1, pady=20)

        # Create a "Save Directory" button
        self.save_button = ttk.Button(root, text="Save Directory", command=self.save_directory)
        self.save_button.grid(row=1, column=2, pady=20)

        # Create a "Plot time series" checkbox
        self.t_serie_var = IntVar()
        self.plot_time_series = ttk.Checkbutton(root, text="Plot time series", variable=self.t_serie_var)
        self.plot_time_series.config(state="disabled")
        self.plot_time_series.grid(row=2, column=1, columnspan=2)

        # Create a "Plot integrated series" checkbox
        self.i_serie_var = IntVar()
        self.plot_integrated_series = ttk.Checkbutton(root, text="Plot integrated series", variable=self.i_serie_var)
        self.plot_integrated_series.config(state="disabled")
        self.plot_integrated_series.grid(row=3, column=1, columnspan=2)

        # Create a "Plot epochs" checkbox
        self.epochs_var = IntVar()
        self.plot_epochs = ttk.Checkbutton(root, text="Plot epochs", variable=self.epochs_var)
        self.plot_epochs.config(state="disabled")
        self.plot_epochs.grid(row=4, column=1, columnspan=2)

        # Create an "Initial Window size" entry
        self.window_size_label = ttk.Label(root, text="Initial Window Size", foreground="gray30")

        self.window_size_label.grid(row=5, column=1, pady=(40, 0))

        self.window_size = ttk.Entry(root, width=12, foreground="gray30")
        self.window_size.insert(0, "10")
        self.window_size.grid(row=5, column=2, padx=(10, 10), pady=(40, 0))
        self.window_size.config(state="disabled")

        self.window_size_max = ttk.Label(root, text="Maximum ", font=("Aptos", 10), foreground="gray30")
        self.window_size_max.grid(row=6, column=2, pady=(0, 20))

        # Create a "Window step" entry
        self.window_step_label = ttk.Label(root, text="Window Step", foreground="gray30")
        self.window_step_label.grid(row=7, column=1, pady=(20, 0))

        self.window_step = ttk.Entry(root, width=12, foreground="gray30")
        self.window_step.insert(0, "45")
        self.window_step.grid(row=7, column=2, padx=(10, 10), pady=(20, 0))
        self.window_step.config(state="disabled")

        # Create a "Plot" button
        self.plot_button = ttk.Button(root, text="Plot", command=self.plot_data)
        self.plot_button.config(state="disabled")
        self.plot_button.grid(row=8, column=1, columnspan=2, pady=(40, 0))

        # Create a "Progress bar" widget
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", maximum=100.1)
        self.progress.grid(row=9, column=1, columnspan=2, pady=(20, 0))

        self.progress_label = ttk.Label(root, text="- %", foreground="SteelBlue2")
        self.progress_label.grid(row=10, column=1, columnspan=2)

    def import_file_data(self) -> torch.Tensor:
        self.txt_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text files", "*.txt")])
        if self.txt_path:
            # Process the selected file (you can replace this with your own logic)
            txt_file = open(self.txt_path, "r")

            self.file_name = self.txt_path.rsplit("/", 1)[-1].split(".")[0]

            # THE FOLLOWING CODE IS JUST FOR DELFINITLI PROJECT
            Time, RAW = np.loadtxt(txt_file, skiprows=1, dtype={'names': ('Time', 'RAW'), 'formats': ('<f8', '<f8')}, unpack=True, usecols=(0, 2))

            txt_file.close()

            np_x, np_y = Time, RAW

            self.x = torch.from_numpy(np_x)
            self.y = torch.from_numpy(np_y)

            self._ready_to_plot()

    def save_directory(self):
        self.save_path = filedialog.askdirectory(title='Select a directory to save the plots')

        self._ready_to_plot()

    def plot_data(self):
        Thread(target=DFA.DFA_F_Plot, args=(self.x, self.y, int(self.window_size.get()), int(self.window_step.get()),
                                            self.epochs_var.get(), self.t_serie_var.get(), self.i_serie_var.get(),
                                            self.save_path, self.hook, self.file_name)).start()

    def hook(self, i):
        self.progress["value"] = i
        self.progress_label.config(text=str(int(i)) + " %")

    def _ready_to_plot(self):
        if self.save_path and self.txt_path:
            self.plot_button.config(state="enabled")
            self.plot_time_series.config(state="enabled")
            self.plot_integrated_series.config(state="enabled")
            self.plot_epochs.config(state="enabled")
            self.window_size.config(state="enabled", foreground="gray80")
            self.window_size_max.config(text="Maximum " + str(round(self.x.size(0) / 4)), foreground="IndianRed3")
            self.window_size_label.config(foreground="gray80")
            self.window_size.bind("<FocusIn>", self._temp_text)

            self.window_step.config(state="enabled", foreground="gray80")
            self.window_step_label.config(foreground="gray80")
            self.window_step.bind("<FocusIn>", self._temp_text)

    @staticmethod
    def _temp_text(event):
        event.widget.delete(0, "end")


if __name__ == "__main__":

    root = Tk()
    root.option_add("*Font", "Aptos")

    # Setting window dimensions
    root.geometry("500x600")

    root.grid_columnconfigure((0, 3), weight=1)

    sv_ttk.set_theme("dark")

    # Setting app title
    root.title("DFA Tool")

    app = App(root)

    # Mainloop to run application
    root.mainloop()
