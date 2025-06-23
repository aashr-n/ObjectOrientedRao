##Pulse onset/start to next pulse start
#change so that you can store/reuse data

import tkinter as tk
from tkinter import filedialog
import os

import mne
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.interpolate # Added for spline interpolation
import matplotlib.gridspec as gridspec # For more control over subplots
from mne.time_frequency import psd_array_multitaper
from scipy.signal import find_peaks

# --- Configuration Switch for Testing ---
USE_DEFAULT_TEST_PATHS = True # Set to False to use file dialogs

# --- Define Default Paths (EDIT THESE FOR YOUR SYSTEM) ---
if USE_DEFAULT_TEST_PATHS:
    DEFAULT_EEG_FILE_PATH = "/Users/aashray/Documents/ChangLab/RCS04_2443.edf" # Replace with your actual .fif or .edf path
    DEFAULT_NPZ_FILE_PATH = "/Users/aashray/Documents/ChangLab/RCS04_2443_psds.npz" # Replace with your actual .npz path (associated with the EEG file)

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

if USE_DEFAULT_TEST_PATHS:
    eeg_file_path = DEFAULT_EEG_FILE_PATH
    npz_file_path = DEFAULT_NPZ_FILE_PATH
    print(f"--- USING DEFAULT TEST PATHS ---")
    print(f"Default EEG file: {eeg_file_path}")
    print(f"Default .npz file: {npz_file_path}")
    if not os.path.exists(eeg_file_path):
        print(f"Error: Default EEG file not found at '{eeg_file_path}'. Please check the path or set USE_DEFAULT_TEST_PATHS to False.")
        raise SystemExit
    if not os.path.exists(npz_file_path):
        print(f"Error: Default .npz file not found at '{npz_file_path}'. Please check the path or set USE_DEFAULT_TEST_PATHS to False.")
        raise SystemExit
else:
    # Step 1: Select the EEG file (for metadata and raw data)
    eeg_file_path = filedialog.askopenfilename(
        title="Select the EEG file (.fif or .edf) (for raw data and metadata)",
        filetypes=[("EEG files", "*.fif *.edf"), ("FIF files", "*.fif"), ("EDF files", "*.edf"), ("All files", "*.*")]
        )
    if not eeg_file_path:
        print("No EEG file was selected. Exiting the program.")
        raise SystemExit

    # Step 2: Select the pre-calculated PSD .npz file
    # Suggest a default .npz filename based on the selected EEG file
    base_eeg_filename = os.path.splitext(os.path.basename(eeg_file_path))[0]
    suggested_npz_filename = f"{base_eeg_filename}_psds.npz"
    initial_dir_npz = os.path.dirname(eeg_file_path)

    npz_file_path = filedialog.askopenfilename(
        title="Select the pre-calculated PSD .npz file",
        filetypes=[("NumPy archive files", "*.npz"), ("All files", "*.*")],
        initialdir=initial_dir_npz,
        initialfile=suggested_npz_filename
    )
    if not npz_file_path:
        print("No .npz file was selected. Exiting the program.")
        raise SystemExit

print(f"Loading EEG data from: {eeg_file_path}")
file_extension = os.path.splitext(eeg_file_path)[1].lower()

if file_extension == '.fif':
    raw = mne.io.read_raw_fif(eeg_file_path, preload=True)
elif file_extension == '.edf':
    raw = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose='WARNING')
else:
    print(f"Unsupported EEG file type: {file_extension}. Please select a .fif or .edf file.")
    raise SystemExit

print(f"Loading PSD data from: {npz_file_path}")
try:
    psd_data_loaded = np.load(npz_file_path)
    psds = psd_data_loaded['psds']
    freqs = psd_data_loaded['freqs']
    print("PSDs and frequencies loaded successfully from .npz file.")
    if psds.shape[0] != len(raw.ch_names):
        print(f"Warning: Mismatch in channel count between .npz file ({psds.shape[0]}) "
              f"and EEG file ({len(raw.ch_names)}). Ensure compatibility.")
    if freqs.ndim != 1:
        print(f"Warning: 'freqs' array from .npz is not 1-dimensional. This might cause issues.")
except KeyError as e:
    print(f"Error: The .npz file '{npz_file_path}' is missing a required key: {e}. Expected 'psds' and 'freqs'.")
    raise SystemExit
except Exception as e:
    print(f"Error loading .npz file '{npz_file_path}': {e}")
    raise SystemExit

# Parameters derived from the loaded EEG file

sampleRate = raw.info['sfreq']
data = raw.get_data()                # shape (n_chan, n_samples)


n_channels = len(raw.ch_names)
n_chan_eeg, _ = data.shape # Number of channels from EEG file
n_chan_psd = psds.shape[0]   # Number of channels from loaded PSDs

print(f"The EEG recording has {n_chan_eeg} channels. Loaded PSDs have {n_chan_psd} channels.")
if n_chan_eeg != n_chan_psd:
    print("WARNING: Channel count mismatch may affect channel-specific analyses if channel names/order differ.")
# These parameters were for PSD calculation, which is now skipped.
# They are kept commented in case any downstream code might have implicitly used them,
# but they are not used for loading PSDs.
# frequencyRes = 0.25
# n_fft = int(sampleRate / frequencyRes)
# bandwidth = 0.1 # This was the bandwidth for psd_array_multitaper

# The PSD calculation block below is now replaced by loading from the .npz file.
# --- (Original PSD calculation block removed) ---


def find_prominent_peaks(
    psd_values, 
    frequencies, 
    prominence=1, 
    distance=None, 
    **kwargs):
    """
    Finds prominent peaks in a Power Spectral Density (PSD) plot.

    This function is a wrapper around scipy.signal.find_peaks, designed to
    identify local maxima that are prominent relative to their surroundings.

    Args:
        psd_values (np.ndarray): The values of the PSD (the y-axis).
        frequencies (np.ndarray): The corresponding frequency values (the x-axis).
        prominence (float, optional): The required prominence of peaks. A peak's
            prominence is the vertical distance between its tip and the lowest
            contour line that encircles it but no higher peak. Defaults to 1.
        distance (int, optional): The minimum required horizontal distance (in
            number of samples) between neighboring peaks. Defaults to None.
        **kwargs: Other keyword arguments to be passed directly to
                  scipy.signal.find_peaks (e.g., 'height', 'width', 'threshold').

    Returns:
        float or None: The lowest frequency (in Hz) of a peak with relative
                       prominence > 5. Returns None if no such peak is found.
    """
    print("Finding prominent peaks and calculating relative prominences...")
    # Use scipy's find_peaks to locate the indices of the peaks
    # The 'prominence' parameter is key to finding peaks relative to their surroundings
    peak_indices, properties = find_peaks(
        psd_values,
        prominence=prominence,
        distance=distance,
        **kwargs
    )

    # If no peaks are found, return an empty dictionary
    if len(peak_indices) == 0:
        print("No initial peaks found by scipy.signal.find_peaks.")
        return None

    # Extract the frequencies and PSD values at the peak indices
    peak_frequencies = frequencies[peak_indices]
    peak_psd_values = psd_values[peak_indices]
    peak_prominences = properties['prominences']
    
    # Calculate relative prominence for each peak
    relative_prominences_list = []
    for i, peak_idx in enumerate(peak_indices):
        current_peak_power = psd_values[peak_idx]
        current_freq = frequencies[peak_idx]
        
        # Dynamically calculate neighborhood_hz for this peak
        # 1/5th of its frequency, capped at a maximum of 1.0 Hz
        neighborhood_hz = min(current_freq / 5.0, 1.0) #####
        lower_bound = current_freq - neighborhood_hz
        upper_bound = current_freq + neighborhood_hz
        
        # Create a mask to select frequencies in the neighborhood, excluding the peak itself
        neighborhood_mask = (frequencies >= lower_bound) & (frequencies <= upper_bound) & (frequencies != current_freq)
        
        neighborhood_power_values = psd_values[neighborhood_mask]
        
        rel_prom_val = 0.0 # Default if neighborhood is empty or too small
        if len(neighborhood_power_values) > 0:
            mean_neighborhood_power = np.mean(neighborhood_power_values)
            if mean_neighborhood_power > 0: # Avoid division by zero
                rel_prom_val = current_peak_power / mean_neighborhood_power
        relative_prominences_list.append(rel_prom_val)

    relative_prominences_arr = np.array(relative_prominences_list)

    # Filter peaks with relative prominence > "5.0"
    relativePromThreshold = 5.0  #################################################change if needed!!
    strong_peak_mask = relative_prominences_arr > relativePromThreshold

    strong_peak_freqs = peak_frequencies[strong_peak_mask]
    strong_peak_psd_values = peak_psd_values[strong_peak_mask]
    strong_peak_std_proms = peak_prominences[strong_peak_mask]
    strong_peak_rel_proms = relative_prominences_arr[strong_peak_mask]

    print(f"\n--- Peaks with Relative Prominence > {relativePromThreshold:.2f} ---")
    if len(strong_peak_freqs) > 0:
        # Sort by frequency for consistent output and selection of lowest
        sort_indices = np.argsort(strong_peak_freqs)
        strong_peak_freqs = strong_peak_freqs[sort_indices]
        strong_peak_psd_values = strong_peak_psd_values[sort_indices]
        strong_peak_std_proms = strong_peak_std_proms[sort_indices]
        strong_peak_rel_proms = strong_peak_rel_proms[sort_indices]

        for i in range(len(strong_peak_freqs)):
            print(f"  Peak at {strong_peak_freqs[i]:.2f} Hz, PSD: {strong_peak_psd_values[i]:.2e}, "
                  f"Std Prom: {strong_peak_std_proms[i]:.2f}, Rel Prom: {strong_peak_rel_proms[i]:.2f}")
        
        lowest_freq_strong_peak = strong_peak_freqs[0] # Already sorted by frequency
        print(f"Returning lowest frequency with relative prominence > {relativePromThreshold:.2f}: {lowest_freq_strong_peak:.2f} Hz")
        return lowest_freq_strong_peak
    else:
        print(f"No peaks found with relative prominence > {relativePromThreshold:.2f}.")
        return None

# Helper function to calculate relative prominence for a single peak
def calculate_single_peak_relative_prominence(psd_values, frequencies, peak_frequency, neighborhood_hz_rule="dynamic"):
    """
    Calculates relative prominence for a specific peak frequency in a PSD.

    Args:
        psd_values (np.ndarray): 1D PSD array.
        frequencies (np.ndarray): Corresponding 1D frequency array.
        peak_frequency (float): The frequency of the peak to analyze.
        neighborhood_hz_rule (str or float): Rule for neighborhood.
            "dynamic": 1/5th of peak_frequency, capped at 1.0 Hz.
            float: Fixed Hz value for neighborhood.

    Returns:
        float: The relative prominence of the peak.
    """
    peak_idx = np.argmin(np.abs(frequencies - peak_frequency))
    current_peak_power = psd_values[peak_idx]
    current_freq = frequencies[peak_idx] # Use the actual frequency from the array

    if neighborhood_hz_rule == "dynamic":
        neighborhood_hz = min(current_freq / 5.0, 1.0)
    elif isinstance(neighborhood_hz_rule, (int, float)):
        neighborhood_hz = neighborhood_hz_rule
    else:
        neighborhood_hz = 1.0 # Default if rule is not recognized

    lower_bound = current_freq - neighborhood_hz
    upper_bound = current_freq + neighborhood_hz

    neighborhood_mask = (frequencies >= lower_bound) & (frequencies <= upper_bound) & (frequencies != current_freq)
    neighborhood_power_values = psd_values[neighborhood_mask]

    rel_prom_val = 0.0
    if len(neighborhood_power_values) > 0:
        mean_neighborhood_power = np.mean(neighborhood_power_values)
        if mean_neighborhood_power > 1e-10: # Avoid division by zero or near-zero
            rel_prom_val = current_peak_power / mean_neighborhood_power
    return rel_prom_val


# --- Main Execution Block (Corrected) ---

def plot_comparison_psds(
    freqs_array,
    mean_psd_array,
    strong_channel_psd_array,
    strong_channel_name,
    estimated_stim_freq=None
):
    """
    Plots the mean PSD and the PSD of the strongest channel for comparison.

    Args:
        freqs_array (np.ndarray): Array of frequencies.
        mean_psd_array (np.ndarray): PSD values for the mean across channels.
        strong_channel_psd_array (np.ndarray): PSD values for the strongest channel.
        strong_channel_name (str): Name of the strongest channel.
        estimated_stim_freq (float, optional): Estimated stimulation frequency to mark.
    """
    plt.figure(figsize=(12, 7))
    
    if estimated_stim_freq is not None: # Changed line style and color
        # Draw the vertical line first to put it in the background
        plt.axvline(estimated_stim_freq, color='green', linestyle='-', alpha=0.6, linewidth=1.5, label=f'Est. Stim Freq: {estimated_stim_freq:.2f} Hz')

    # Plot PSDs on top of the vertical line
    plt.plot(freqs_array, 10 * np.log10(mean_psd_array), label='Mean PSD across all channels', alpha=0.8)
    if strong_channel_psd_array is not None and strong_channel_name:
        plt.plot(freqs_array, 10 * np.log10(strong_channel_psd_array), label=f'PSD of Channel: {strong_channel_name}', alpha=0.8)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title('Comparison of Mean PSD and Strongest Channel PSD')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlim(max(0, freqs_array.min()), min(freqs_array.max(), 150)) # Limit x-axis for better viz, e.g., up to 150 Hz
    plt.tight_layout()
    # plt.show() # Will be called once in main


# --- New Modular Functions for Dog Line Analysis ---

def create_dog_line(channel_data, peak_detection_height=0):
    """
    Creates a "Dog Line" from the positive peaks of the third derivative of a signal.

    Args:
        channel_data (np.ndarray): The 1D raw data for a single channel.
        peak_detection_height (float): The minimum height for detecting peaks on the
            third derivative. A value of 0 will find all positive peaks.

    Returns:
        tuple: (dog_line, third_derivative, peak_indices) or (None, np.ndarray, None) if failed.
    """
    # 1. Calculate the third derivative
    third_derivative = np.diff(channel_data, n=3)
    
    # 2. Find all positive peaks on the third derivative
    # Using a low height threshold to catch all positive peaks as requested.
    peak_indices, _ = find_peaks(third_derivative, height=peak_detection_height)
    
    if len(peak_indices) < 2:
        print("Warning: Less than 2 positive peaks found on derivative; cannot create dog line.")
        return None, third_derivative, peak_indices

    # 3. Perform linear interpolation and ensure the "Dog Line" is non-negative
    peak_values = third_derivative[peak_indices]
    interp_func = scipy.interpolate.interp1d(
        peak_indices, peak_values, kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    x_full_derivative = np.arange(len(third_derivative))
    dog_line_raw = interp_func(x_full_derivative)
    dog_line = np.maximum(dog_line_raw, 0) # Clip at zero

    return dog_line, third_derivative, peak_indices

def find_pulses_on_dog_line(
    dog_line,
    third_derivative,
    onset_percentile=70,
    middle_percentile=98,
    closing_percentile=70
):
    """
    Detects pulses on a pre-computed "Dog Line" using a 3-threshold state machine.
    Thresholds are based on percentiles of the absolute third derivative values.

    Args:
        dog_line (np.ndarray): The pre-computed, non-negative dog line signal.
        third_derivative (np.ndarray): The third derivative signal, used for thresholding.
        onset_percentile (float): Percentile of abs(third_derivative) for the onset threshold.
        middle_percentile (float): Percentile of abs(third_derivative) for the middle threshold.
        closing_percentile (float): Percentile of abs(third_derivative) for the closing threshold.

    Returns:
        tuple: (pulse_starts, pulse_ends, thresholds)
    """
    # 1. Define thresholds based on percentiles of the absolute third derivative
    abs_third_deriv = np.abs(third_derivative)
    onset_threshold = np.percentile(abs_third_deriv, onset_percentile)
    middle_threshold = np.percentile(abs_third_deriv, middle_percentile)
    closing_threshold = np.percentile(abs_third_deriv, closing_percentile)

    thresholds = {
        'onset': onset_threshold,
        'middle': middle_threshold,
        'closing': closing_threshold
    }

    # 2. State machine for pulse detection
    state = "IDLE"  # States: IDLE, ARMED (onset crossed), CONFIRMED (middle crossed)
    pulse_starts, pulse_ends = [], []
    current_pulse_start = None

    for i, value in enumerate(dog_line):
        if state == "IDLE":
            if value > onset_threshold:
                state = "ARMED"
                current_pulse_start = i
        elif state == "ARMED":
            if value > middle_threshold:
                state = "CONFIRMED"
            elif value < closing_threshold:
                state = "IDLE"
                current_pulse_start = None
        elif state == "CONFIRMED":
            if value < closing_threshold:
                pulse_starts.append(current_pulse_start)
                pulse_ends.append(i)
                state = "IDLE"
                current_pulse_start = None

    if state == "CONFIRMED" and current_pulse_start is not None:
        pulse_starts.append(current_pulse_start)
        pulse_ends.append(len(dog_line) - 1)

    return pulse_starts, pulse_ends, thresholds

def plot_dog_line_analysis(
    raw_signal, times_raw, channel_name,
    third_derivative, dog_line, peak_indices_on_deriv,
    pulse_starts, pulse_ends, thresholds
):
    """Plots the results of the dog line analysis."""
    plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

    # Top plot: Overview of the original signal for context
    ax0 = plt.subplot(gs[0])
    ax0.plot(times_raw, raw_signal, color='purple', alpha=0.7)
    ax0.set_title(f'Original Signal: Channel {channel_name}')
    ax0.set_ylabel('Amplitude')
    ax0.grid(True, alpha=0.5)
    ax0.set_xlim(times_raw.min(), times_raw.max())

    # Bottom plot: Derivative, Dog Line, and Pulse Detection
    ax1 = plt.subplot(gs[1])
    x_axis_samples = np.arange(len(third_derivative))
    ax1.plot(x_axis_samples, third_derivative, label='Third Derivative', alpha=0.5, color='gray')
    ax1.plot(x_axis_samples, dog_line, label='Dog Line', linestyle='-', color='dodgerblue', linewidth=2.0)
    # ax1.plot(peak_indices_on_deriv, third_derivative[peak_indices_on_deriv], 'x', color='black', markersize=5, label=f'Derivative Peaks ({len(peak_indices_on_deriv)})')

    # Plot thresholds (positive only, since dog line is non-negative)
    ax1.axhline(thresholds['onset'], color='green', linestyle=':', label=f'Onset Thresh ({thresholds["onset"]:.1e})')
    ax1.axhline(thresholds['middle'], color='orange', linestyle=':', label=f'Middle Thresh ({thresholds["middle"]:.1e})')
    ax1.axhline(thresholds['closing'], color='red', linestyle=':', label=f'Closing Thresh ({thresholds["closing"]:.1e})')

    # Highlight detected pulses using axvspan
    if pulse_starts:
        ax1.axvspan(pulse_starts[0], pulse_ends[0], color='red', alpha=0.2, label=f'Detected Pulses ({len(pulse_starts)})')
        for i in range(1, len(pulse_starts)):
            ax1.axvspan(pulse_starts[i], pulse_ends[i], color='red', alpha=0.2)

    ax1.set_title(f'Dog Line Pulse Detection for Channel {channel_name}')
    ax1.set_xlabel('Sample Number (of derivative signal)')
    ax1.set_ylabel('Amplitude (d³/dt³)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.5)
    ax1.set_xlim(0, len(third_derivative))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

def merge_close_pulses(starts, ends, min_separation_samples):
    """
    Merges detected pulses that are closer than a specified minimum separation.

    Args:
        starts (list or np.ndarray): List of pulse start indices.
        ends (list or np.ndarray): List of pulse end indices.
        min_separation_samples (int): The minimum number of samples between the end
            of one pulse and the start of the next to keep them separate.

    Returns:
        tuple: (merged_starts, merged_ends)
    """
    if len(starts) < 2:
        return starts, ends

    merged_starts = [starts[0]]
    merged_ends = [ends[0]]

    for i in range(1, len(starts)):
        gap = starts[i] - merged_ends[-1]
        if gap < min_separation_samples:
            merged_ends[-1] = ends[i] # Extend the previous pulse
        else:
            merged_starts.append(starts[i])
            merged_ends.append(ends[i])
    return merged_starts, merged_ends

def remove_stim_artifacts_spline(raw_channel_data, pulse_starts_samples):
    """
    Removes stimulation artifacts from a channel's time series by performing
    linear interpolation between the start of each consecutive detected pulse.

    Args:
        raw_channel_data (np.ndarray): The 1D raw data for a single channel.
        pulse_starts_samples (list): List of start sample indices of detected pulses.

    Returns:
        np.ndarray: The signal with artifacts interpolated.
    """
    interpolated_signal = np.copy(raw_channel_data)
    
    print(f"\n--- Performing linear interpolation between consecutive pulse starts ---")

    if not pulse_starts_samples or len(pulse_starts_samples) < 2:
        print("Not enough pulse starts (< 2) for interpolation. Returning original signal.")
        return interpolated_signal

    # Iterate through pairs of consecutive pulse starts
    for i in range(len(pulse_starts_samples) - 1):
        start_sample1 = pulse_starts_samples[i]
        start_sample2 = pulse_starts_samples[i+1]

        # The known points are the two consecutive pulse starts
        x_known = np.array([start_sample1, start_sample2])
        y_known = raw_channel_data[x_known]

        # The region to fill is the interval between them (inclusive of the second point)
        x_to_fill = np.arange(start_sample1, start_sample2 + 1)
        
        # Create the interpolation function
        try:
            f_interp = scipy.interpolate.interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value='extrapolate')
        except ValueError as e:
            print(f"Warning: Could not create interpolation for samples {start_sample1}-{start_sample2}: {e}. Skipping this segment.")
            continue
        
        # Perform interpolation
        interpolated_values = f_interp(x_to_fill)
        
        # Apply interpolated values to the signal copy
        interpolated_signal[x_to_fill] = interpolated_values
    
    print("Interpolation between pulse starts complete.")
    return interpolated_signal

def plot_artifact_removal(original_signal, interpolated_signal, times, normal_pulses, long_pulses, channel_name):
    """Plots the original and interpolated signals, highlighting the removed artifact regions."""
    plt.figure(figsize=(18, 8))
    plt.plot(times, original_signal, label='Original Signal', color='blue', alpha=0.7)
    plt.plot(times, interpolated_signal, label='Interpolated Signal', color='red', alpha=0.8, linestyle='--')

    # Plot normal pulses in gray
    if normal_pulses:
        for i, (start, end) in enumerate(normal_pulses):
            start_time = times[start]
            end_time = times[end]
            if i == 0:
                plt.axvspan(start_time, end_time, color='gray', alpha=0.3, label=f'Interpolated Regions ({len(normal_pulses)})')
            else:
                plt.axvspan(start_time, end_time, color='gray', alpha=0.3)

    # Plot the top 1% longest pulses in a different color for debugging
    if long_pulses:
        for i, (start, end) in enumerate(long_pulses):
            start_time = times[start]
            end_time = times[end]
            if i == 0:
                plt.axvspan(start_time, end_time, color='orange', alpha=0.4, label=f'Top 1% Longest Pulses ({len(long_pulses)})')
            else:
                plt.axvspan(start_time, end_time, color='orange', alpha=0.4)

    plt.title(f'Stimulation Artifact Removal using Spline Interpolation (Channel: {channel_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xlim(times.min(), times.max())
    plt.tight_layout()

def plot_spectrogram_comparison(original_signal, cleaned_signal, sfreq, channel_name, stim_freq=None):
    """Plots before and after spectrograms for artifact removal comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle(f'Spectrogram Comparison for Artifact Removal (Channel: {channel_name})', fontsize=16)

    # Calculate spectrogram for the original signal
    f_orig, t_orig, Sxx_orig = signal.spectrogram(original_signal, fs=sfreq, nperseg=int(sfreq))
    pcm1 = ax1.pcolormesh(t_orig, f_orig, 10 * np.log10(Sxx_orig + np.finfo(float).eps), shading='gouraud', cmap='viridis')
    ax1.set_title('Before Artifact Removal')
    ax1.set_ylabel('Frequency (Hz)')
    # if stim_freq:
    #     ax1.axhline(stim_freq, color='red', linestyle='--', alpha=0.7, label=f'Stim Freq ({stim_freq:.1f} Hz)')
    # ax1.legend(loc='upper right')
    fig.colorbar(pcm1, ax=ax1, label='Power/Frequency (dB/Hz)')

    # Calculate spectrogram for the cleaned signal, using the same color scale for fair comparison
    f_clean, t_clean, Sxx_clean = signal.spectrogram(cleaned_signal, fs=sfreq, nperseg=int(sfreq))
    pcm2 = ax2.pcolormesh(t_clean, f_clean, 10 * np.log10(Sxx_clean + np.finfo(float).eps), shading='gouraud', cmap='viridis', vmin=pcm1.get_clim()[0], vmax=pcm1.get_clim()[1])
    ax2.set_title('After Artifact Removal')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    # if stim_freq:
    #     ax2.axhline(stim_freq, color='red', linestyle='--', alpha=0.7)
    fig.colorbar(pcm2, ax=ax2, label='Power/Frequency (dB/Hz)')

    plt.ylim(0, stim_freq * 2 if stim_freq and stim_freq > 0 else 150) # Focus on relevant frequencies
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_psd_comparison(original_signal, cleaned_signal, sfreq, channel_name, stim_freq=None):
    """Plots before and after Power Spectral Densities (PSDs) for comparison."""
    plt.figure(figsize=(12, 7))
    
    # Calculate PSDs using multitaper for consistency
    psd_orig, freqs = psd_array_multitaper(original_signal[np.newaxis, :], sfreq=sfreq, fmin=0, fmax=sfreq/2, verbose=False)
    psd_clean, _ = psd_array_multitaper(cleaned_signal[np.newaxis, :], sfreq=sfreq, fmin=0, fmax=sfreq/2, verbose=False)

    # Plotting
    plt.plot(freqs, 10 * np.log10(psd_orig[0]), label='Original PSD', color='blue', alpha=0.8)
    plt.plot(freqs, 10 * np.log10(psd_clean[0]), label='Cleaned PSD (after interpolation)', color='red', alpha=0.8, linestyle='-')

    if stim_freq:
        plt.axvline(stim_freq, color='green', linestyle='--', alpha=0.7, label=f'Stim Freq ({stim_freq:.1f} Hz)')

    plt.title(f'PSD Comparison Before and After Artifact Removal (Channel: {channel_name})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlim(0, 150) # Limit to a reasonable frequency range for visualization
    plt.tight_layout()

def plot_individual_artifact_segments(raw_channel_data, interpolated_channel_data, pulse_starts_samples, sfreq, channel_name, stim_freq=None, context_ms=50, max_plots=10, num_harmonics_to_notch=3):
    """
    Plots individual artifact segments, comparing the signal and PSD before and after interpolation.

    Args:
        raw_channel_data (np.ndarray): The original 1D data for the channel.
        interpolated_channel_data (np.ndarray): The 1D data after interpolation.
        pulse_starts_samples (list): List of start samples for each artifact.
        sfreq (float): Sampling frequency.
        channel_name (str): Name of the channel for plot titles.
        stim_freq (float, optional): The estimated stimulation frequency. Used to define a
            high-pass filter to add original high-frequency content back to the
            interpolated signal for visualization. Defaults to None.
        num_harmonics_to_notch (int): The number of harmonics (including the fundamental)
            to remove with notch filters for the comparison plot.
        context_ms (float): Time in ms to show before and after the artifact for context.
        max_plots (int): The maximum number of segment plots to generate to avoid creating too many figures.
    """
    if not pulse_starts_samples or len(pulse_starts_samples) < 2:
        print("Not enough pulses to plot individual segments.")
        return

    print(f"\n--- Generating plots for individual artifact segments to compare before/after interpolation ---")
    context_samples = int(context_ms / 1000 * sfreq)
    times_full = np.arange(len(raw_channel_data)) / sfreq

    # Pre-calculate a notch-filtered version of the signal for comparison.
    # This shows an alternative artifact removal method (filtering vs. interpolation).
    notch_filtered_signal_full = None
    if stim_freq is not None and stim_freq > 0:
        print(f"Pre-calculating notch-filtered signal (base freq: {stim_freq:.1f} Hz) for comparison plot.")
        notch_filtered_signal_full = np.copy(raw_channel_data)
        nyquist = 0.5 * sfreq
        
        # Apply notch filter for the fundamental and its harmonics
        for i in range(1, num_harmonics_to_notch + 1):
            freq_to_notch = stim_freq * i
            if freq_to_notch < nyquist:
                Q = 30.0  # Quality factor, determines notch width. Higher Q = narrower notch.
                b, a = signal.iirnotch(freq_to_notch, fs=sfreq, Q=Q)
                notch_filtered_signal_full = signal.filtfilt(b, a, notch_filtered_signal_full)

    num_segments = len(pulse_starts_samples) - 1
    
    if num_segments > max_plots:
        print(f"Warning: {num_segments} artifact segments found. "
              f"Displaying a dispersed subset of {max_plots} plots to avoid excessive memory usage.")
        # Select evenly spaced indices to plot, ensuring we cover the start, middle, and end
        indices_to_plot = np.linspace(0, num_segments - 1, num=max_plots, dtype=int)
    else:
        indices_to_plot = range(num_segments)

    for i in indices_to_plot:
        start_sample = pulse_starts_samples[i]
        end_sample = pulse_starts_samples[i+1]

        plot_start = max(0, start_sample - context_samples)
        plot_end = min(len(raw_channel_data), end_sample + 1 + context_samples)

        # Get data segments for both original and interpolated signals
        segment_times = times_full[plot_start:plot_end]
        original_segment_data = raw_channel_data[plot_start:plot_end]
        interpolated_segment_data = interpolated_channel_data[plot_start:plot_end]
        
        # Get the core artifact data for PSD calculation
        artifact_data_orig = raw_channel_data[start_sample:end_sample+1]
        artifact_data_interp = interpolated_channel_data[start_sample:end_sample+1]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Artifact Segment {i+1} Comparison for Channel: {channel_name}', fontsize=16)

        # --- Time Series Plot ---
        ax1.plot(segment_times, original_segment_data, label='Original Signal', color='blue', alpha=0.7)
        ax1.plot(segment_times, interpolated_segment_data, label='Interpolated Signal', color='red', linestyle='-', alpha=0.8)

        # If the notch-filtered signal was created, use it to add "texture" back to the interpolated line.
        if notch_filtered_signal_full is not None:
            # 1. Get the segment of the notch-filtered signal. This is our best estimate of the true signal
            #    with the main artifact frequencies removed.
            notch_filtered_segment = notch_filtered_signal_full[plot_start:plot_end]

            # 2. To get the "texture", we high-pass filter the notched segment to remove its own low-frequency trend.
            #    A low cutoff (e.g., 1 Hz) is suitable for isolating texture from baseline.
            hp_cutoff = 1.0  # Hz
            nyquist = 0.5 * sfreq
            # Ensure segment is long enough for a stable filter application (e.g., > 100ms)
            if hp_cutoff < nyquist and len(notch_filtered_segment) > sfreq * 0.1: 
                b, a = signal.butter(4, hp_cutoff / nyquist, btype='high', analog=False)
                texture = signal.filtfilt(b, a, notch_filtered_segment)
                
                # 3. Add this texture back onto the smooth, flat interpolated signal.
                recombined_signal = interpolated_segment_data + texture
                
                ax1.plot(segment_times, recombined_signal, label='Interpolated + Recovered Texture', color='green', linestyle='-', alpha=0.9)
            else:
                # Fallback for short segments: just plot the notch-filtered signal directly.
                ax1.plot(segment_times, notch_filtered_segment, label='Notch Filtered Signal (Fallback)', color='green', linestyle='--', alpha=0.9)

        ax1.axvspan(times_full[start_sample], times_full[end_sample], color='orange', alpha=0.2, label='Interpolated Region')
        ax1.set_title(f'Time Series (Samples {start_sample} to {end_sample})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # --- PSD Plot ---
        ax2.set_title('PSD Comparison of Highlighted Region (Before vs. After)')
        
        # Plot PSD for original artifact data
        if len(artifact_data_orig) > 0:
            psd_orig, freqs_segment = psd_array_multitaper(artifact_data_orig[np.newaxis, :], sfreq=sfreq, fmin=0, fmax=sfreq/2, verbose=False)
            ax2.plot(freqs_segment, 10 * np.log10(psd_orig[0]), color='blue', alpha=0.8, label='Original PSD')
        
        # Plot PSD for interpolated artifact data
        if len(artifact_data_interp) > 0:
            psd_interp, freqs_segment_interp = psd_array_multitaper(artifact_data_interp[np.newaxis, :], sfreq=sfreq, fmin=0, fmax=sfreq/2, verbose=False)
            ax2.plot(freqs_segment_interp, 10 * np.log10(psd_interp[0]), color='red', alpha=0.8, label='Interpolated PSD')

        if len(artifact_data_orig) > 0 or len(artifact_data_interp) > 0:
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Power (dB/Hz)')
            ax2.grid(True, which="both", ls="-", alpha=0.5)
            ax2.set_xlim(0, 150)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Segment too short for PSD', ha='center', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Main Execution Block (Corrected) ---
if __name__ == '__main__':
    # Ensure Matplotlib backend is suitable for interactive plots if not running in a specific environment
    # For example, if running from a simple script, 'Qt5Agg' or 'TkAgg' might be needed.
    # plt.switch_backend('Qt5Agg') # Or 'TkAgg', etc. Uncomment if plots are not interactive.
    try:
        # 1. Calculate Mean PSD and Initial Frequency Estimate
        mean_psd = psds.mean(axis=0)
        avg_signal_full = np.mean(data, axis=0)
        
        estStimFrequency = find_prominent_peaks(
            psd_values=mean_psd,
            frequencies=freqs,
            prominence=0.2,
            distance=50
        )
        # 2. Determine Strongest Channel based on Band Power and Refine Frequency
        final_est_stim_freq = estStimFrequency # Initialize with estimate from mean PSD
        strong_channel_idx_for_viz = None

        if estStimFrequency is not None:
            print(f"\nInitial estimated stimulation frequency (from mean PSD): {estStimFrequency:.2f} Hz")

            # New logic for selecting strong_channel_idx_for_viz based on band power
            target_freq_for_band = estStimFrequency
            if target_freq_for_band > 0:
                band_total_width = 1.0 / (target_freq_for_band * 10.0)
                band_half_width = band_total_width / 2.0
                
                lower_b = target_freq_for_band - band_half_width
                upper_b = target_freq_for_band + band_half_width

                # Ensure bounds are within the actual frequency range available and logical
                lower_b = max(lower_b, freqs.min())
                upper_b = min(upper_b, freqs.max())

                if upper_b > lower_b: # Check if the band is valid
                    freq_mask = (freqs >= lower_b) & (freqs <= upper_b)
                    
                    if np.any(freq_mask):
                        # Calculate sum of PSD values in the band for each channel
                        channel_band_powers = np.sum(psds[:, freq_mask], axis=1)
                        
                        if len(channel_band_powers) > 0 and n_channels > 0:
                            strong_channel_idx = np.argmax(channel_band_powers)
                            strong_channel_idx_for_viz = strong_channel_idx
                            print(f"Channel with strongest power in band [{lower_b:.2f}-{upper_b:.2f} Hz] "
                                  f"around {target_freq_for_band:.2f} Hz: {raw.ch_names[strong_channel_idx_for_viz]} "
                                  f"(Power: {channel_band_powers[strong_channel_idx]:.2e})")

                            # Refine final_est_stim_freq using this selected strong channel's PSD
                            refined_freq_from_strong_channel = find_prominent_peaks(
                                psd_values=psds[strong_channel_idx_for_viz, :],
                                frequencies=freqs,
                                prominence=0.1, # Parameters for refinement
                                distance=20
                            )
                            if refined_freq_from_strong_channel is not None:
                                final_est_stim_freq = refined_freq_from_strong_channel
                                # This print will be covered by the one after the if block
                            # else, final_est_stim_freq remains estStimFrequency (already set)
                        else:
                            print("Warning: Could not calculate band powers (e.g., no channels or empty PSDs). Using initial frequency estimate.")
                    else:
                        print(f"Warning: Frequency band [{lower_b:.2f}-{upper_b:.2f} Hz] is empty or outside data range. "
                              "Using initial frequency estimate.")
                else: # upper_b <= lower_b
                    print(f"Warning: Invalid frequency band calculated (lower: {lower_b:.2f}, upper: {upper_b:.2f} Hz). Using initial frequency estimate.")
            else:
                print(f"Warning: Initial estimated frequency ({target_freq_for_band:.2f} Hz) is not positive. "
                      "Cannot define band. Using initial frequency estimate.")
        
        if final_est_stim_freq is not None:
            print(f"Final estimated stimulation frequency to be used: {final_est_stim_freq:.2f} Hz")
        else:
            print("No stimulation frequency could be determined.")

        # Plot the comparison PSDs if data is available
        if 'mean_psd' in locals() and 'freqs' in locals():
            psd_strong_ch_for_plot = None
            ch_name_strong_for_plot = "N/A"
            if strong_channel_idx_for_viz is not None and psds is not None:
                psd_strong_ch_for_plot = psds[strong_channel_idx_for_viz, :]
                ch_name_strong_for_plot = raw.ch_names[strong_channel_idx_for_viz]
            plot_comparison_psds(
                freqs_array=freqs,
                mean_psd_array=mean_psd,
                strong_channel_psd_array=psd_strong_ch_for_plot,
                strong_channel_name=ch_name_strong_for_plot,
                estimated_stim_freq=final_est_stim_freq)

        # --- New Analysis: Third Derivative and "Dog Line" ---
        if strong_channel_idx_for_viz is not None:
            print("\n--- Performing Dog Line Pulse Detection on Strongest Channel ---")
            strong_channel_name = raw.ch_names[strong_channel_idx_for_viz]
            strong_channel_data = data[strong_channel_idx_for_viz, :]

            # Step 1: Create the dog line, finding all positive peaks
            dog_line, third_deriv, deriv_peaks = create_dog_line(
                channel_data=strong_channel_data,
                peak_detection_height=0  # Find all positive peaks, no matter how small
            )

            # Step 2: Find pulses on the dog line and plot results
            if dog_line is not None:
                # Find initial pulses based on new percentile thresholds
                initial_pulse_starts, initial_pulse_ends, thresholds = find_pulses_on_dog_line(
                    dog_line, third_deriv
                )

                print(f"Dog Line Thresholds (based on 3rd deriv percentiles):")
                print(f"  Onset: {thresholds['onset']:.2e}, Middle: {thresholds['middle']:.2e}, Closing: {thresholds['closing']:.2e}")

                # Step 3: Merge close pulses
                if initial_pulse_starts and final_est_stim_freq is not None and final_est_stim_freq > 0:
                    min_dist_sec = 1.0 / (10 * final_est_stim_freq)
                    min_dist_samples = int(min_dist_sec * sampleRate)
                    print(f"Merging pulses closer than {min_dist_sec*1000:.1f} ms ({min_dist_samples} samples)...")
                    
                    merged_starts, merged_ends = merge_close_pulses(
                        initial_pulse_starts, initial_pulse_ends, min_dist_samples
                    )
                    print(f"Found {len(initial_pulse_starts)} initial pulses, merged into {len(merged_starts)} final pulses.")
                else:
                    merged_starts, merged_ends = initial_pulse_starts, initial_pulse_ends
                    if not initial_pulse_starts:
                        print("No pulses met the threshold criteria on the dog line.")
                    else:
                        print("Skipping pulse merging (no stim freq available).")

                # Step 4: Perform spline interpolation to remove artifacts
                interpolated_strong_channel_data = remove_stim_artifacts_spline(
                    raw_channel_data=strong_channel_data,
                    pulse_starts_samples=merged_starts
                )

                # Plot individual artifact segments to compare before and after interpolation
                plot_individual_artifact_segments(
                    raw_channel_data=strong_channel_data,
                    interpolated_channel_data=interpolated_strong_channel_data,
                    pulse_starts_samples=merged_starts,
                    sfreq=sampleRate,
                    channel_name=strong_channel_name,
                    stim_freq=final_est_stim_freq
                )

                # Step 5: Separate pulses by duration for plotting and then graph the results
                normal_pulses = []
                long_pulses = []
                if len(merged_starts) > 1: # Need at least two starts to have a region between them
                    # The interpolated regions are the gaps between consecutive pulse starts
                    interp_region_starts = merged_starts[:-1]
                    interp_region_ends = merged_starts[1:]

                    pulse_durations = np.array(interp_region_ends) - np.array(interp_region_starts)
                    
                    if len(pulse_durations) > 0:
                        # Define the threshold for the top 1% longest pulses
                        duration_threshold = np.percentile(pulse_durations, 99)
                        print(f"Highlighting interpolated regions longer than the 99th percentile duration ({duration_threshold:.0f} samples).")

                        for start, end, duration in zip(interp_region_starts, interp_region_ends, pulse_durations):
                            if duration >= duration_threshold:
                                long_pulses.append((start, end))
                            else:
                                normal_pulses.append((start, end))
                else: # Not enough pulses to form regions, so both lists are empty
                    pass

                # --- Plotting for Verification ---
                # Plot Spectrogram Comparison
                plot_spectrogram_comparison(
                    original_signal=strong_channel_data,
                    cleaned_signal=interpolated_strong_channel_data,
                    sfreq=sampleRate,
                    channel_name=strong_channel_name,
                    stim_freq=final_est_stim_freq
                )
                # Plot PSD Comparison
                plot_psd_comparison(
                    original_signal=strong_channel_data,
                    cleaned_signal=interpolated_strong_channel_data,
                    sfreq=sampleRate,
                    channel_name=strong_channel_name,
                    stim_freq=final_est_stim_freq
                )

                plot_artifact_removal(
                    original_signal=strong_channel_data,
                    interpolated_signal=interpolated_strong_channel_data,
                    times=raw.times,
                    normal_pulses=normal_pulses,
                    long_pulses=long_pulses,
                    channel_name=strong_channel_name
                )

                plot_dog_line_analysis(
                    raw_signal=strong_channel_data,
                    times_raw=raw.times,
                    channel_name=strong_channel_name,
                    third_derivative=third_deriv,
                    dog_line=dog_line,
                    peak_indices_on_deriv=deriv_peaks,
                    pulse_starts=merged_starts,
                    pulse_ends=merged_ends,
                    thresholds=thresholds
                )
            else:
                print("Analysis skipped as dog line could not be created.")
        else:
            print("\nSkipping third derivative analysis because no strong channel was identified.")
        
        plt.show() # Display all generated figures

    except SystemExit:
        print("Program execution cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")