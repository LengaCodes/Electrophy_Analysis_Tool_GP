import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
import pywt

# === Paths and Parameters ===
folder_path = r'D:\example'
rec_name = 'example'
filename = os.path.join(folder_path, f'{rec_name}.h5')
dataset_name = 'Raw-0.0'
col_idx = 1

# Time and frequency params
fs = 30000  # Original sampling rate
start_time_sec = 0
end_time_sec = 600
start_idx = int(start_time_sec * fs)
end_idx = int(end_time_sec * fs)

# Filter bands
broad_band = (1, 8000)
low_band = (1, 80)
notch_freq = 50.0
quality_factor = 30.0

# Downsampling for TFP
dwn_fac = 300  # reduces to 100 Hz
vmax = 20  # dB scale max

# === Load data ===
with h5py.File(filename, 'r') as file:
    raw_data = np.array(file[dataset_name])
ch_data = raw_data[:, col_idx]
data_segment = ch_data[start_idx:end_idx]


# === Signal Filtering Function ===
def filter_signal(data, fs, band, notch_freq=50.0, quality_factor=30.0):
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    data_notch = scipy.signal.filtfilt(b_notch, a_notch, data)
    b_band, a_band = scipy.signal.butter(2, band, btype='bandpass', fs=fs)
    return scipy.signal.filtfilt(b_band, a_band, data_notch)


# === Get Scales for TFP ===
def get_scales_for_band(wavelet, band, fs, num=500):
    min_freq, max_freq = band
    cf = pywt.central_frequency(wavelet)
    return np.linspace(cf * fs / max_freq, cf * fs / min_freq, num=num)


# === Filtering ===
broad_filtered = filter_signal(data_segment, fs, broad_band)
low_filtered = filter_signal(data_segment, fs, low_band)

# === Downsampling for TFP ===
low_filtered_dwn = scipy.signal.decimate(low_filtered, dwn_fac)
fs_lfp = fs // dwn_fac

# === Time Axes ===
time_sec_full = np.linspace(start_time_sec, end_time_sec, len(broad_filtered))      # full-band
time_sec_low = np.linspace(start_time_sec, end_time_sec, len(low_filtered))         # low-band
time_sec_tfp = np.linspace(start_time_sec, end_time_sec, len(low_filtered_dwn))     # TFP

# === Compute TFP ===
scales = get_scales_for_band('morl', low_band, fs_lfp)
frequencies = pywt.scale2frequency('morl', scales) * fs_lfp
coef, _ = pywt.cwt(low_filtered_dwn, scales, 'morl', 1.0 / fs_lfp)
TF_power = np.abs(coef) ** 2
baseline_power = np.median(TF_power, axis=1, keepdims=True)
TF_power_db = 10 * np.log10(TF_power / baseline_power + 1e-12)

# Downsample time axis for plotting TFP
downsample_factor = 10
TF_time = time_sec_tfp[::downsample_factor]
TF_power_db_ds = TF_power_db[:, ::downsample_factor]
X, Y = np.meshgrid(TF_time, frequencies)

# === Plotting with Gridspec ===
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1.3], wspace=0.05, hspace=0.3)

# Full-band trace
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(time_sec_full, broad_filtered, color='black', linewidth=0.5)
ax0.set_ylabel("Amplitude (µV)")
ax0.set_title(f"Full band signal 1-8000 Hz")
ax0.set_ylim(-250, 250)
ax0.set_xlim(start_time_sec, end_time_sec)
ax0.grid(False)
ax0.tick_params(labelbottom=False)  # Hide x-axis labels

# Low-frequency trace
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(time_sec_low, low_filtered, color='black', linewidth=0.5)
ax1.set_ylabel("Amplitude (µV)")
ax1.set_title(f"Low band signal 1-80 Hz")
ax1.set_ylim(-50, 50)
ax1.set_xlim(start_time_sec, end_time_sec)
ax1.grid(False)
ax1.tick_params(labelbottom=False)  # Hide x-axis labels

# TFP Plot
ax2 = fig.add_subplot(gs[2, 0])
c = ax2.pcolormesh(X, Y, TF_power_db_ds, shading='gouraud', cmap='viridis', vmin=0, vmax=vmax)
ax2.set_ylabel("Frequency (Hz)")
ax2.set_title(f"Time frequency plot")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(low_band)
ax2.set_xlim(start_time_sec, end_time_sec)

# Colorbar
cbar_ax = fig.add_subplot(gs[2, 1])
fig.colorbar(c, cax=cbar_ax, label="Power (dB)")

plt.tight_layout()
plt.show()
