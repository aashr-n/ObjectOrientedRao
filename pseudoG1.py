import tkinter as tk
from tkinter import filedialog
import os

import mne
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper

# --- 1. GUI to Select File (and get initial parameters) ---
def get_fif_file():
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
        title="Select an EEG .fif file",
        filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("No file was selected. Exiting the program.")
        raise SystemExit
        
    print(f"Loading selected file: {file_path}")
    # Load the data into memory for processing
    raw = mne.io.read_raw_fif(file_path, preload=True)
    return raw

# --- 2. Calculate PSD for All Channels ---
def calculate_all_psds(raw):
    """
    Calculates the Power Spectral Density (PSD) for every channel in the Raw object
    using the multitaper method.

    Args:
        raw (mne.io.Raw): The loaded MNE Raw object.

    Returns:
        tuple: A tuple containing:
            - psds (np.ndarray): A 2D array of PSDs (n_channels x n_freqs).
            - freqs (np.ndarray): A 1D array of corresponding frequencies.
    """
    print("Calculating PSD for all channels using the multitaper method...")
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Using psd_array_multitaper for high-quality spectral estimation
    psds, freqs = psd_array_multitaper(
        data,
        sfreq=sfreq,
        fmin=1,  # Start at 1 Hz to avoid DC offset issues
        fmax=sfreq / 2,
        adaptive=True,  # Let the method optimize the weights
        low_bias=True,
        verbose=False
    )
    print("PSD calculation complete.")
    return psds, freqs

# --- 3. Average All PSDs ---
def average_psds(psds):
    """
    Calculates the mean PSD across all channels.

    Args:
        psds (np.ndarray): A 2D array of PSDs (n_channels x n_freqs).

    Returns:
        np.ndarray: A 1D array representing the mean PSD.
    """
    print("Averaging PSDs across all channels.")
    mean_psd = np.mean(psds, axis=0)
    return mean_psd

# --- 4. Find Stim Frequency by Relative Prominence ---
def find_stim_freq_by_relative_prominence(mean_psd, freqs, neighborhood_hz=5.0):
    """
    Finds the frequency with the highest relative prominence in a PSD.

    Relative prominence is defined as the peak's power divided by the average
    power of its immediate frequency neighborhood, making it a measure of
    how much a peak "stands out".

    Args:
        mean_psd (np.ndarray): The 1D averaged PSD array.
        freqs (np.ndarray): The corresponding frequency array.
        neighborhood_hz (float): The width (in Hz) on each side of a peak to
                                 define its local neighborhood for comparison.

    Returns:
        float: The frequency in Hz with the highest relative prominence.
    """
    print(f"Searching for peaks with highest relative prominence...")
    
    # First, find all significant peaks using standard prominence
    # The prominence is the vertical distance between the peak and its lowest contour line
    peaks, properties = signal.find_peaks(mean_psd, prominence=(np.max(mean_psd) / 20))
    
    if len(peaks) == 0:
        print("Could not find any significant peaks.")
        return None

    relative_prominences = []
    peak_freqs = freqs[peaks]

    # Calculate relative prominence for each peak
    for i, peak_idx in enumerate(peaks):
        peak_power = mean_psd[peak_idx]
        current_freq = freqs[peak_idx]
        
        # Define the local neighborhood to compare against
        lower_bound = current_freq - neighborhood_hz
        upper_bound = current_freq + neighborhood_hz
        
        # Create a mask to select frequencies in the neighborhood, excluding the peak itself
        neighborhood_mask = (freqs >= lower_bound) & (freqs <= upper_bound) & (freqs != current_freq)
        
        # Get the power of the surrounding frequencies
        neighborhood_power = mean_psd[neighborhood_mask]
        
        if len(neighborhood_power) == 0:
            # If the neighborhood is empty, this peak is likely noise or at the edge
            relative_prominences.append(0)
            continue
            
        # Calculate the mean power of the neighborhood
        mean_neighborhood_power = np.mean(neighborhood_power)
        
        # Relative prominence is the ratio of the peak's power to its surroundings
        relative_prominence = peak_power / mean_neighborhood_power
        relative_prominences.append(relative_prominence)

    # Sort peaks by their calculated relative prominence
    sorted_indices = np.argsort(relative_prominences)[::-1]
    
    print("\n--- Top 5 Peaks by Relative Prominence (Sanity Check) ---")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        freq = peak_freqs[idx]
        rel_prom = relative_prominences[idx]
        print(f"  {i+1}. Frequency: {freq:.2f} Hz, Relative Prominence Score: {rel_prom:.2f}")
    print("-----------------------------------------------------------\n")

    # The best candidate for the stimulation frequency is the one that stands out the most
    best_peak_index = sorted_indices[0]
    stim_freq_hz = peak_freqs[best_peak_index]
    
    print(f"Identified stimulation frequency: {stim_freq_hz:.2f} Hz")
    return stim_freq_hz

# --- 5. Visualize Stimulation Epoch with a Sliding Window ---
def visualize_stim_epoch_with_sliding_window(raw, stim_freq_hz, window_sec=1.0, step_sec=0.1, threshold_factor=1.5):
    """
    Uses a sliding window to find when the stimulation frequency is most prominent
    and visualizes the result on a spectrogram.

    Args:
        raw (mne.io.Raw): The loaded MNE Raw object.
        stim_freq_hz (float): The stimulation frequency to track.
        window_sec (float): The duration of the sliding window in seconds.
        step_sec (float): The amount to slide the window forward each step, in seconds.
        threshold_factor (float): Multiplier for the median relative prominence to set
                                  the detection threshold.
    """
    print("Analyzing recording with a sliding window to find the stimulation epoch...")
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    n_channels, n_samples = data.shape
    
    # Convert window and step from seconds to samples
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    
    window_starts = np.arange(0, n_samples - window_samples, step_samples)
    
    prominence_over_time = []

    # Slide the window across the data
    for start in window_starts:
        end = start + window_samples
        window_data = data[:, start:end]
        
        # Calculate PSD for this short window
        psds, freqs = psd_array_multitaper(window_data, sfreq=sfreq, fmin=1, fmax=sfreq/2, verbose=False)
        mean_psd = np.mean(psds, axis=0)
        
        # Find the frequency bin closest to our stimulation frequency
        stim_freq_idx = np.argmin(np.abs(freqs - stim_freq_hz))
        peak_power = mean_psd[stim_freq_idx]
        
        # Define neighborhood for relative prominence calculation
        neighborhood_hz = 5.0
        lower_bound = freqs[stim_freq_idx] - neighborhood_hz
        upper_bound = freqs[stim_freq_idx] + neighborhood_hz
        neighborhood_mask = (freqs >= lower_bound) & (freqs <= upper_bound) & (freqs != freqs[stim_freq_idx])
        
        mean_neighborhood_power = np.mean(mean_psd[neighborhood_mask]) if np.any(neighborhood_mask) else 1e-10
        
        relative_prominence = peak_power / mean_neighborhood_power
        prominence_over_time.append(relative_prominence)
    
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f'Stimulation Epoch Analysis (Detected Stim Freq: {stim_freq_hz:.2f} Hz)', fontsize=16)

    # Plot 1: Relative Prominence over Time
    window_times = window_starts / sfreq
    ax1.plot(window_times, prominence_over_time, color='dodgerblue', label='Relative Prominence Score')
    
    # Calculate and plot a detection threshold
    detection_threshold = np.median(prominence_over_time) * threshold_factor
    ax1.axhline(detection_threshold, color='red', linestyle='--', label=f'Detection Threshold')
    ax1.set_ylabel('Relative Prominence Score')
    ax1.set_title('Stimulation Frequency Prominence Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # Plot 2: Spectrogram of the average signal
    avg_signal = np.mean(data, axis=0)
    f, t, Sxx = signal.spectrogram(avg_signal, fs=sfreq, nperseg=int(sfreq))
    
    # Use a logarithmic scale for power for better visualization
    pcm = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + np.finfo(float).eps), shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Spectrogram with Detected Stimulation Epoch')
    ax2.set_ylim(0, stim_freq_hz * 3) # Focus on relevant frequencies
    fig.colorbar(pcm, ax=ax2, label='Power/Frequency (dB/Hz)')
    
    # Highlight the detected epoch on the spectrogram
    active_stim_times = window_times[np.array(prominence_over_time) > detection_threshold]
    if len(active_stim_times) > 0:
        start_time = active_stim_times[0]
        end_time = active_stim_times[-1] + window_sec # Add window duration to the last start time
        ax2.axvspan(start_time, end_time, color='orangered', alpha=0.3, label='Detected Stim Epoch')

    ax2.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        # 1. Get file from user
        raw_file = get_fif_file()
        
        # 2. Calculate all PSDs
        all_channel_psds, frequencies = calculate_all_psds(raw_file)
        
        # 3. Average the PSDs
        mean_psd_across_channels = average_psds(all_channel_psds)
        
        # 4. Find the most likely stimulation frequency
        estimated_stim_freq = find_stim_freq_by_relative_prominence(mean_psd_across_channels, frequencies)
        
        if estimated_stim_freq is not None:
            # 5. Visualize the active stimulation epoch using the found frequency
            visualize_stim_epoch_with_sliding_window(raw_file, estimated_stim_freq)
            
    except SystemExit:
        # This catch block prevents a traceback error if the user closes the file dialog.
        print("Program execution cancelled by user.")

