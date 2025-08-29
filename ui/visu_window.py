import tkinter as tk
from tkinter import ttk  
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .base_window import BaseWindow
from .result_window import ResultWindow
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.signal import decimate

class VisuWindow(BaseWindow):
    Ordered_channels = [f'Raw-0.{i}' for i in range(32)]

    def __init__(self, file_path, icon_path=None, image_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = file_path  # Store the file path

        # Get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set window size to a percentage of the screen size 
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.8)

        self.geometry(f"{window_width}x{window_height}")

        # Frame for plot
        self.plot_frame = tk.Frame(self.inner_frame, bg='white', width=int(window_width * 0.85))
        self.plot_frame.pack(side='left', fill='both', expand=True)
        self.plot_frame.pack_propagate(0)

        # Frame for checkboxes and 'Begin Analysis' button
        self.selection_frame = tk.Frame(self.inner_frame, bg='white', width=int(window_width * 0.15))
        self.selection_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        self.selection_frame.pack_propagate(0)

        # Title for checkboxes section
        title_label = tk.Label(self.selection_frame, text="Electrodes to Analyze", font=("Arial", 12), bg='white')
        title_label.grid(row=0, column=0, columnspan=4, pady=(10, 15))

        self.checkbox_vars = []

        # Checkboxes for electrodes
        for group_num in range(1, 5):
            organo_label = tk.Label(self.selection_frame, text=f"Organoide {group_num}", font=("Arial", 10), bg='white')
            organo_label.grid(row=(group_num - 1) * 6 + 1, column=0, columnspan=4, pady=4)

            for i in range(8):
                electrode_number = (group_num - 1) * 8 + i + 1
                if electrode_number <= len(self.Ordered_channels):
                    var = tk.IntVar(value=1)
                    checkbox = tk.Checkbutton(self.selection_frame, text=f'{electrode_number}', variable=var, bg='white', command=self.update_next_button_click)
                    checkbox.grid(row=(group_num - 1) * 7 + 2 + i // 4, column=i % 4, sticky='w')
                    self.checkbox_vars.append(var)

            spacer_label = tk.Label(self.selection_frame, text="", bg='white')
            spacer_label.grid(row=(group_num - 1) * 7 + 4, column=0, columnspan=4, pady=23)

        # Begin Analysis button
        self.begin_button = tk.Button(self.selection_frame, text="Begin Analysis", font=("Arial", 12), bg='lightgray', command=self.open_check_window, state=tk.DISABLED)
        self.begin_button.grid(row=25, column=0, columnspan=4, pady=10)

        # Plot the raw signals
        self.plot_signals(file_path)

    def open_check_window(self):
        selected_electrodes = [f'Raw-0.{i}' for i, var in enumerate(self.checkbox_vars) if var.get()]
        elec_number = [f"{i + 1}" for i, var in enumerate(self.checkbox_vars) if var.get()]
        self.destroy()
        result = ResultWindow(self.file_path, selected_electrodes, elec_number)
        result.mainloop()

    def update_next_button_click(self):
        if any(var.get() for var in self.checkbox_vars):
            self.begin_button.config(state=tk.NORMAL)
        else:
            self.begin_button.config(state=tk.DISABLED)

    def plot_signals(self, file_path):
        progress_bar = ttk.Progressbar(self.plot_frame, orient='horizontal', mode='determinate', length=400)
        progress_bar.pack(pady=10)
        progress_bar['maximum'] = len(self.Ordered_channels)

        fig, axes = plt.subplots(4, 8, figsize=(18, 8))
        axes = axes.flatten()

        with h5py.File(file_path, 'r') as file:
            fs = 300
            total_time = 600
            window_duration = 2
            window_size = int(fs * window_duration)
            amplitude_threshold = 35
            downsample_factor = 30

            for i, name in enumerate(self.Ordered_channels):
                try:
                    dataset = file[name]
                    data = np.array(dataset)

                    if data.size == 0:
                        continue

                    if data.ndim > 1:
                        data = data[:, 1]

                    time_vector = np.arange(data.shape[0]) / fs

                    noise_segment = None
                    noise_start_index = None
                    for j in range(0, len(data) - window_size, window_size):
                        window = data[j:j + window_size]
                        if np.max(np.abs(window)) <= amplitude_threshold:
                            noise_segment = window
                            noise_start_index = j
                            break

                    threshold = None
                    if noise_segment is not None:
                        noise_std = np.std(noise_segment)
                        threshold = 6 * noise_std

                    detected_activities = []
                    refractory_window_samples = int(0.005 * fs)
                    last_activity_time = -refractory_window_samples
                    if threshold:
                        for k in range(1, len(data)):
                            if abs(data[k]) > threshold and (k > last_activity_time + refractory_window_samples):
                                detected_activities.append(k)
                                last_activity_time = k

                    if len(data) >= downsample_factor:
                        plot_data = decimate(data, downsample_factor, zero_phase=True)
                        plot_time = np.linspace(0, len(data) / fs, len(plot_data))
                    else:
                        plot_data = data
                        plot_time = time_vector
                    
                    axes[i].plot(plot_time, plot_data, color='black', linewidth=0.5)

                    if noise_segment is not None and noise_start_index is not None:
                        noise_start_time = noise_start_index / fs
                        noise_end_time = (noise_start_index + window_size) / fs
                        axes[i].axvspan(noise_start_time, noise_end_time, color='yellow', alpha=0.3)

                    if detected_activities:
                        activity_times = np.array(detected_activities) / fs
                        activity_amplitudes = data[detected_activities]
                        axes[i].scatter(activity_times, activity_amplitudes, color='red', s=5)

                    if threshold:
                        axes[i].axhline(y=threshold, color='green', linestyle='--', linewidth=0.5)
                        axes[i].axhline(y=-threshold, color='green', linestyle='--', linewidth=0.5)

                    axes[i].set_title(f'Electrode {i + 1}', fontsize=8)

                except KeyError:
                    print(f"Dataset '{name}' is missing.")
                except Exception as e:
                    print(f"Error accessing dataset '{name}': {e}")

                progress_bar['value'] = i + 1
                self.update_idletasks()

            progress_bar.destroy()

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
