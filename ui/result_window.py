import tkinter as tk
from .base_window import BaseWindow
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
from tkinter import filedialog, messagebox
import threading
from unittest.mock import patch

# -------------------------
# Helper Function: process_analyze
# -------------------------
def process_analyze(file_path, electrode, config, num):
    print(f"Starting analysis for electrode {num} ({electrode})")
    # Load data
    with h5py.File(file_path, 'r') as file:
        raw_data = np.array(file[electrode])
    ch2_data = raw_data[:, 1]  # Extract channel 2 data

    # Filter functions
    def apply_notch_filter(data, fs, notch_freq=50.0, quality_factor=30.0):
        b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
        return scipy.signal.filtfilt(b_notch, a_notch, data)
    
    def apply_bandpass_filter(data, fs, low_cutoff, high_cutoff, order=2):
        b_band, a_band = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs)
        return scipy.signal.filtfilt(b_band, a_band, data)
    
    # Noise-free segment detection
    def find_noise_free_segment(data, threshold, duration_samples, sampling_rate):
        max_iterations = len(data) - duration_samples + 1
        for start_idx in range(max_iterations):
            segment = data[start_idx:start_idx + duration_samples]
            if np.all(np.abs(segment) <= threshold):
                start_time = start_idx / sampling_rate
                end_time = (start_idx + duration_samples) / sampling_rate
                print(f"Found noise-free segment from {start_time} to {end_time} for electrode {num}")
                return pd.DataFrame([{"Start Time (s)": start_time, "End Time (s)": end_time}])
        print(f"No noise-free segment found for electrode {num} after {max_iterations} iterations")
        return None

    # Activity detection
    def detect_activities(filtered_data, threshold, sampling_rate):
        peaks = []
        for k in range(1, len(filtered_data) - 1):
            if (
                abs(filtered_data[k]) > threshold and
                abs(filtered_data[k]) > abs(filtered_data[k - 1]) and
                abs(filtered_data[k]) > abs(filtered_data[k + 1])
            ):
                peaks.append({"Time (s)": k / sampling_rate, "Amplitude": filtered_data[k]})
        return pd.DataFrame(peaks)

    # Burst processing functions
    def detect_bursts(peaks_df, time_condition, min_spike):
        if peaks_df.empty or "Time (s)" not in peaks_df.columns:
            return pd.DataFrame()
        activity = peaks_df['Time (s)']
        debTime, finTime, spikeCount = 0, 0, 0
        bursts = []
        for spike, next_spike in zip(activity[:-1], activity[1:]):
            if debTime == 0 and abs(spike - next_spike) < time_condition:
                debTime = spike
                spikeCount = 1
            elif debTime != 0 and abs(spike - next_spike) < time_condition:
                spikeCount += 1
            elif debTime != 0 and abs(spike - next_spike) > time_condition:
                finTime = spike
                if spikeCount >= min_spike:
                    bursts.append({'begTime': debTime, 'endTime': finTime})
                debTime, finTime, spikeCount = 0, 0, 0
        return pd.DataFrame(bursts)
    
    def merge_bursts(bursts_df, time_condition):
        merged_bursts = []
        current_burst = {"begTime": bursts_df.iloc[0]["begTime"], "endTime": bursts_df.iloc[0]["endTime"]}
        for burst_end, next_burst_start, next_burst_end in zip(
            bursts_df["endTime"][:-1], bursts_df["begTime"][1:], bursts_df["endTime"][1:]):
            if abs(burst_end - next_burst_start) < time_condition:
                current_burst["endTime"] = next_burst_end
            else:
                merged_bursts.append(current_burst)
                current_burst = {"begTime": next_burst_start, "endTime": next_burst_end}
        merged_bursts.append(current_burst)
        return pd.DataFrame(merged_bursts)
    
    def extract_burst_activities(peaks_df, burst_df):
        if peaks_df.empty or burst_df.empty:
            return pd.DataFrame()
        burst_activities = []
        for _, burst_row in burst_df.iterrows():
            burst_start = burst_row["begTime"]
            burst_end = burst_row["endTime"]
            burst_data = peaks_df[(peaks_df["Time (s)"] >= burst_start) & (peaks_df["Time (s)"] <= burst_end)]
            burst_activities.append(burst_data)
        return pd.concat(burst_activities, ignore_index=True)

    # Main workflow
    sampling_rate = config["sampling_rate"]
    window_size = int(config["window_duration_s"] * sampling_rate)

    # Apply filters
    notch_filtered_data = apply_notch_filter(ch2_data, sampling_rate)
    filtered_LF = apply_bandpass_filter(notch_filtered_data, sampling_rate, 1, 40)
    filtered_HF = apply_bandpass_filter(notch_filtered_data, sampling_rate, 500, 8000)

    # Compute PSD using Welch's method
    segment_length = 2 * sampling_rate  # 2-second segments
    frequencies, psd_values = scipy.signal.welch(notch_filtered_data, fs=sampling_rate, nperseg=segment_length)
    psd_values *= 1e6  # Convert from V²/Hz to mV²/Hz

    # Define frequency bands and compute mean power in each
    bands = {
        "Delta 0.5-4 [Hz]": (0.5, 4),
        "Theta 4-8 [Hz]": (4, 8),
        "Alpha 8-13 [Hz]": (8, 13),
        "Beta 13-30 [Hz]": (13, 30),
        "Gamma 30-80 [Hz]": (30, 80),
        "FO 80-250 [Hz]": (80, 250),
        "vFO 250-500 [Hz]": (250, 500),
        "AP 500-8000 [Hz]": (500, 8000),
    }

    band_powers = {}
    for band, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies <= high)
        mean_power = np.mean(psd_values[mask])
        band_powers[band] = mean_power
    PSD_df = pd.DataFrame([band_powers])
    PSD_df.insert(0, "Electrode", num)

    # Detect noise-free segments
    noise_free_segment = find_noise_free_segment(notch_filtered_data, config["amplitude_threshold"], window_size, sampling_rate)

    lf_elec_data = []
    hf_elec_data = []

    if noise_free_segment is not None:
        start_time = noise_free_segment.iloc[0]["Start Time (s)"]
        end_time = noise_free_segment.iloc[0]["End Time (s)"]
        start_idx, end_idx = int(start_time * sampling_rate), int(end_time * sampling_rate)
        # Calculate thresholds for activity detection
        mean_LF, std_LF = np.mean(np.abs(filtered_LF[start_idx:end_idx])), np.std(np.abs(filtered_LF[start_idx:end_idx]))
        mean_HF, std_HF = np.mean(np.abs(filtered_HF[start_idx:end_idx])), np.std(np.abs(filtered_HF[start_idx:end_idx]))
        threshold_LF = mean_LF + config["lf_threshold_factor"] * std_LF
        threshold_HF = mean_HF + config["hf_threshold_factor"] * std_HF
        lf_peaks_df = detect_activities(filtered_LF, threshold_LF, sampling_rate)
        hf_peaks_df = detect_activities(filtered_HF, threshold_HF, sampling_rate)
        lf_bursts = detect_bursts(lf_peaks_df, config["lf_time_condition"], config["min_spike"])
        hf_bursts = detect_bursts(hf_peaks_df, config["hf_time_condition"], config["min_spike"])
        if not lf_bursts.empty:
            merged_lf_bursts = merge_bursts(lf_bursts, config["lf_time_condition_burst"])
            lf_burst_activities = extract_burst_activities(lf_peaks_df, merged_lf_bursts)
            merged_lf_bursts['Time Interval'] = merged_lf_bursts['begTime'] - merged_lf_bursts['endTime'].shift()
            merged_lf_bursts['Time burst'] = merged_lf_bursts['endTime'] - merged_lf_bursts['begTime']
            timeBtwMoy = round(merged_lf_bursts['Time Interval'].mean(), 3)
            timeMax = round(merged_lf_bursts['Time Interval'].max(), 3)
            duration = round(merged_lf_bursts['Time burst'].mean(), 3)
            spikeNumber = lf_peaks_df.shape[0]
            burstNumber = merged_lf_bursts.shape[0]
            spikeInBurst = lf_burst_activities.shape[0]

            lf_elec_data.append({
                'Electrodes': num,
                'Mean interval interburst [s]': timeBtwMoy, 
                'Interval interburst maximum [s]': timeMax, 
                'Mean burst duration [s]': duration, 
                'Number of Spike': spikeNumber, 
                'Number of Burst': burstNumber, 
                'Number of spike in burst': spikeInBurst
            })

        if not hf_bursts.empty:
            merged_hf_bursts = merge_bursts(hf_bursts, config["hf_time_condition_burst"])
            hf_burst_activities = extract_burst_activities(hf_peaks_df, merged_hf_bursts)
            merged_hf_bursts['Time Interval'] = merged_hf_bursts['begTime'] - merged_hf_bursts['endTime'].shift()
            merged_hf_bursts['Time burst'] = merged_hf_bursts['endTime'] - merged_hf_bursts['begTime']
            timeBtwMoy = round(merged_hf_bursts['Time Interval'].mean(), 3)
            timeMax = round(merged_hf_bursts['Time Interval'].max(), 3)
            duration = round(merged_hf_bursts['Time burst'].mean(), 3)
            spikeNumber = hf_peaks_df.shape[0]
            burstNumber = merged_hf_bursts.shape[0]
            spikeInBurst = hf_burst_activities.shape[0]

            hf_elec_data.append({
                'Electrodes': num,
                'Mean interval interburst [s]': timeBtwMoy, 
                'Interval interburst maximum [s]': timeMax, 
                'Mean burst duration [s]': duration, 
                'Number of Spike': spikeNumber, 
                'Number of Burst': burstNumber, 
                'Number of spike in burst': spikeInBurst
            })

            burst_latency = merged_hf_bursts[['begTime']].rename(columns={'begTime': num})
    print(f"Completed processing for electrode {num}")
    return lf_elec_data, hf_elec_data, PSD_df, burst_latency

# -------------------------
# ResultWindow Class
# -------------------------
class ResultWindow(BaseWindow):
    def __init__(self, file_path, selected_electrodes, elec_number, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
            # Configuration
        config = {
            "sampling_rate": 30000,  # Hz
            "amplitude_threshold": 35,  # mV
            "window_duration_s": 0.5,  # seconds
            "lf_threshold_factor": 4.5,
            "hf_threshold_factor": 6,
            "refractory_window_ms": 5,  # milliseconds
            "min_spike": 5,
            "lf_time_condition": 0.3,  # seconds
            "hf_time_condition": 0.05,  # seconds
            "lf_time_condition_burst": 2.5,  # seconds
            "hf_time_condition_burst": 0.5,  # seconds
        }
        
        # Global data for all electrodes
        self.lf_global_data = []
        self.hf_global_data = []
        self.PSD_global = []  

        self.organoid1 = pd.DataFrame()
        self.organoid2 = pd.DataFrame()
        self.organoid3 = pd.DataFrame()
        self.organoid4 = pd.DataFrame()
        
        # Flag for analysis completion
        self.analysis_complete = False
        
        # Status label for displaying progress
        self.status_text = tk.StringVar()
        status_label = tk.Label(self.inner_frame, textvariable=self.status_text, font=('Arial', 10), bg='white', fg='green')
        status_label.pack(pady=5)
        self.status_text.set("Status: Beginning of analysis")
        
        # Run analysis in a background thread
        analysis_thread = threading.Thread(target=self.run_analysis, args=(file_path, selected_electrodes, elec_number, config), daemon=True)
        analysis_thread.start()
        
        # Setup export UI
        self.info_label = tk.Label(self.inner_frame, text="Select a folder for exporting the results", font=("Arial", 18), bg="#FFFFFF")
        self.info_label.pack(pady=20)
        self.select_folder = tk.Button(self.inner_frame, text="Folder", command=self.folder_button, font=("Arial", 12))
        self.select_folder.pack(pady=10)
        self.export_label = tk.Label(self.inner_frame, text="", anchor="e", width=20)
        self.export_label.pack(pady=5)
        self.info_label_name = tk.Label(self.inner_frame, text="Enter file name", font=("Arial", 12), bg="#FFFFFF")
        self.info_label_name.pack(pady=10)
        self.file_name = tk.Entry(self.inner_frame, font=("Arial", 12))
        self.file_name.pack(pady=5)
        # Bind event to check file name entry changes
        self.file_name.bind("<KeyRelease>", self.validate_export)
        # Create export button disabled initially
        self.export_button = tk.Button(self.inner_frame, text="Export to Excel", command=self.export_to_excel, font=("Arial", 12), state=tk.DISABLED)
        self.export_button.pack(pady=10)
        
    def validate_export(self, event=None):
        # Enable export button only if analysis is complete and file_name entry is not empty
        if self.analysis_complete and self.file_name.get().strip() != "":
            self.export_button.config(state=tk.NORMAL)
        else:
            self.export_button.config(state=tk.DISABLED)
        
    def update_status(self, num):
        self.status_text.set(f"Status: Completed analysis for electrode {num}")

    def build_mea_adjacency(self):
        base_neighbors = {
            1: {8, 3, 6, 2},
            2: {3, 4, 1},
            3: {1, 2, 4, 6, 5, 8},
            4: {2, 3, 5, 6},
            5: {4, 6, 3, 7},
            6: {3, 5, 7, 8, 4, 1},
            7: {6, 8, 5},
            8: {1, 6, 3, 7},
        }
        adjacency = {}
        for off in (0, 8, 16, 24):
            for base_e, neighs in base_neighbors.items():
                e = base_e + off
                adjacency[e] = {n + off for n in neighs}
        return adjacency
    
    def continuity_score(self, seq, adjacency):
        if not isinstance(seq, (list, tuple)) or len(seq) <= 1:
            return 0.0
        order = [int(x) for x in seq]
        visited = {order[0]}
        hits = 0
        for e in order[1:]:
            if any(e in adjacency.get(v, set()) for v in visited):
                hits += 1
            visited.add(e)
        return hits / (len(order) - 1)

        
    def run_analysis(self, file_path, selected_electrodes, elec_number, config):
        for electrode, num in zip(selected_electrodes, elec_number):
            try:
                results = process_analyze(file_path, electrode, config, num)
            except Exception as e:
                print(f"[ERROR] Failed to process electrode {electrode}: {e}")
                continue  # Skip to the next electrode

            try:
                lf_data, hf_data, psd_data, hf_burst = results

                if lf_data:
                    self.lf_global_data.extend(lf_data)
                else:
                    print(f"[INFO] LF data is empty for electrode {electrode}")

                if hf_data:
                    self.hf_global_data.extend(hf_data)
                else:
                    print(f"[INFO] HF data is empty for electrode {electrode}")

                if psd_data is not None:
                    self.PSD_global.append(psd_data)
                else:
                    print(f"[INFO] PSD data is missing for electrode {electrode}")
                                    
                if hf_burst is not None:
                    num = int(num) 
                    if 1 <= num <= 8:
                        self.organoid1 = pd.concat([self.organoid1, hf_burst], axis=1)
                    elif 9 <= num <= 16:
                        self.organoid2 = pd.concat([self.organoid2, hf_burst], axis=1)
                    elif 17 <= num <= 24:
                        self.organoid3 = pd.concat([self.organoid3, hf_burst], axis=1)
                    elif 25 <= num <= 32:
                        self.organoid4 = pd.concat([self.organoid4, hf_burst], axis=1)

                self.after(0, self.update_status, num)

            except Exception as e:
                print(f"[ERROR] Failed to unpack or process data for electrode {electrode}: {e}")
        
        print("Organoid1 columns:", self.organoid1.columns)
        print("Organoid1 sample data:\n", self.organoid1.head())

        # ---- Burst propagation analysis ----
        def compute_burst_propagation(organoid_burst_start, min_electrodes=3, max_gap=0.150):
            if organoid_burst_start.shape[1] <= 1:
                print("DataFrame must contain more than one column (electrode). Skipping.")
                return pd.DataFrame()

            total_electrodes = len(organoid_burst_start.columns)
            adj_map = self.build_mea_adjacency()

            # Collect all electrode-time pairs
            events = [(t, col) for col in organoid_burst_start.columns 
                            for t in organoid_burst_start[col].dropna().values]
            events.sort(key=lambda x: x[0])

            latency_results = []
            chain, seen = [], set()

            def finalize_chain(chain):
                if len(chain) < min_electrodes:
                    return None

                electrodes_order = [c for _, c in chain]
                times_order = [tt for tt, _ in chain]
                latency_range = times_order[-1] - times_order[0]
                inter_latencies = [t2 - t1 for t1, t2 in zip(times_order[:-1], times_order[1:])]

                return {
                    'electrode_order': electrodes_order,
                    'time_order': times_order,
                    'latency_range': latency_range,
                    'inter_latencies': inter_latencies,
                    'first_electrode': electrodes_order[0],
                    'last_electrode': electrodes_order[-1],
                    'num_electrodes': len(chain),
                    'mean_propagation_time': latency_range / (len(chain) - 1),
                    'num_elec_per': len(chain) / total_electrodes,
                    'continuity_score': self.continuity_score(electrodes_order, adj_map)
                }

            for t, col in events:
                if not chain:
                    chain, seen = [(t, col)], {col}
                    continue

                prev_t, _ = chain[-1]
                if t - prev_t <= max_gap:
                    if col not in seen:  # skip duplicates
                        chain.append((t, col))
                        seen.add(col)
                else:
                    row = finalize_chain(chain)
                    if row:
                        latency_results.append(row)
                    chain, seen = [(t, col)], {col}

            # Finalize last chain
            row = finalize_chain(chain)
            if row:
                latency_results.append(row)

            return pd.DataFrame(latency_results)

        self.lags_org1 = compute_burst_propagation(self.organoid1)
        self.lags_org2 = compute_burst_propagation(self.organoid2)
        self.lags_org3 = compute_burst_propagation(self.organoid3)
        self.lags_org4 = compute_burst_propagation(self.organoid4)

        self.after(0, lambda: self.status_text.set("Analysis complete for all selected electrodes!"))
        self.analysis_complete = True
        self.after(0, self.validate_export)

           
    def folder_button(self):
        self.folder_selected = filedialog.askdirectory()
        if self.folder_selected:
            self.export_label.config(text=os.path.basename(self.folder_selected))
        else:
            self.export_label.config(text="No folder selected")
        # Re-check export conditions (in case folder selection is needed)
        self.validate_export()
    
    def export_to_excel(self):
        file_name = self.file_name.get().strip()
        if not file_name.endswith(".xlsx"):
            file_name += ".xlsx"
        if hasattr(self, 'folder_selected') and self.folder_selected:
            try:
                file_path = os.path.join(self.folder_selected, file_name)
                # Check if PSD data exists before concatenating
                if self.PSD_global:
                    df_PSD = pd.concat(self.PSD_global, ignore_index=True)
                else:
                    expected_cols = ["Electrode", "Delta 0.5-4 [Hz]", "Theta 4-8 [Hz]", "Alpha 8-13 [Hz]",
                                     "Beta 13-30 [Hz]", "Gamma 30-80 [Hz]", "FO 80-250 [Hz]",
                                     "vFO 250-500 [Hz]", "AP 500-8000 [Hz]"]
                    df_PSD = pd.DataFrame(columns=expected_cols)
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    pd.DataFrame(self.lf_global_data).to_excel(writer, sheet_name='LF', index=False)
                    pd.DataFrame(self.hf_global_data).to_excel(writer, sheet_name='HF', index=False)
                    df_PSD.to_excel(writer, sheet_name='PSD', index=False)
                    pd.DataFrame(self.lags_org1).to_excel(writer, sheet_name='O1', index=False)
                    pd.DataFrame(self.lags_org2).to_excel(writer, sheet_name='O2', index=False)
                    pd.DataFrame(self.lags_org3).to_excel(writer, sheet_name='O3', index=False)
                    pd.DataFrame(self.lags_org4).to_excel(writer, sheet_name='O4', index=False)
                messagebox.showinfo("Export Successful", f"Results successfully exported to {file_path}")
                os._exit(0)
            except Exception as e:
                messagebox.showerror("Export Failed", f"An error occurred: {e}")
        else:
            messagebox.showerror("Export Failed", "No folder selected")