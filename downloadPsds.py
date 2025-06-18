#change so that you can store/reuse data

import tkinter as tk
from tkinter import filedialog
import os

import mne
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper
from scipy.signal import find_peaks



# --- 1. GUI to Select File (and get initial parameters) ---
"""
    Opens a GUI for the user to select a .fif file and loads it using MNE.
    
    Returns:
        mne.io.Raw: The loaded MNE Raw object, preloaded into memory.
    """
    # Set up the tkinter root window
tk_root = tk.Tk()
tk_root.withdraw()  # Hide the main window
    
    # MNE logs can be verbose, so we'll only show warnings and errors.
mne.set_log_level('WARNING')

    # Open the file selection dialog
file_path = filedialog.askopenfilename(
    title="Select an EEG file (.fif or .edf)",
    filetypes=[("EEG files", "*.fif *.edf"), ("FIF files", "*.fif"), ("EDF files", "*.edf"), ("All files", "*.*")]
    )
    
if not file_path:
    print("No file was selected. Exiting the program.")
    raise SystemExit        

# Load the data into memory for processing
print(f"Selected file: {file_path}")

file_extension = os.path.splitext(file_path)[1].lower()

if file_extension == '.fif':
    print(f"Loading .fif file: {file_path}")
    raw = mne.io.read_raw_fif(file_path, preload=True)
elif file_extension == '.edf':
    print(f"Loading .edf file: {file_path}")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='WARNING')
else:
    print(f"Unsupported file type: {file_extension}. Please select a .fif or .edf file.")
    raise SystemExit

### Parameters first
#  (gonna straight copy from matlab and then move from there)
#   this would be easier to use if you made a GUI for changing these


sampleRate = raw.info['sfreq']
data = raw.get_data()                # shape (n_chan, n_samples)
n_chan, _ = data.shape

frequencyRes = 0.25 #the divisor defines the
# Increase FFT length for finer frequency resolution (0.01 Hz)
n_fft = int(sampleRate / frequencyRes)


n_channels = len(raw.ch_names)

print(f"The recording has {n_channels} channels.")

# 2) Compute PSD on every channel via multi‐taper

bandwidth = 0.1  # this is W, T is automatically calculated, L is based off T and W
#lower bandwidth  =

# --- Compute PSD per channel with progress updates ---
psd_list = []
for idx, ch in enumerate(raw.ch_names):
    # Updated print statement to overwrite the previous line
    print(f"\rCalculating PSD for channel {idx+1}/{n_channels} ({ch})...{' ' * 20}", end='', flush=True)
    # compute PSD just for this one channel
    psd_ch, freqs = psd_array_multitaper(
        data[idx:idx+1],
        sfreq=sampleRate,
        fmin=0,
        fmax=sampleRate / 2,
        bandwidth=bandwidth,
        adaptive=False,    # or True if you've resolved warnings
        low_bias=True,
        normalization='full',
        verbose=False
    )
    psd_list.append(psd_ch[0])  # grab the 1D PSD array

# Stack them into an array for averaging
psds = np.vstack(psd_list)
# Print a newline after the loop is done to move to the next line for subsequent prints
print("\nMulti-taper PSD computation complete for all channels.")

# --- Save PSDs and Frequencies to a file ---
output_dir = os.path.dirname(file_path)
base_filename = os.path.splitext(os.path.basename(file_path))[0]
output_filename = f"{base_filename}_psds.npz"
output_filepath = os.path.join(output_dir, output_filename)

try:
    np.savez_compressed(output_filepath, psds=psds, freqs=freqs)
    print(f"PSDs and frequencies saved to: {output_filepath}")
except Exception as e:
    print(f"Error saving PSD data: {e}")
