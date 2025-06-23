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
    relative_prominence_threshold=5.0,
    verbose=True,
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
        relative_prominence_threshold (float, optional): The threshold for a peak's
            power relative to its local neighborhood. Defaults to 5.0.
        verbose (bool, optional): If True, prints detailed output. Defaults to True.
        **kwargs: Other keyword arguments to be passed directly to
                  scipy.signal.find_peaks (e.g., 'height', 'width', 'threshold').

    Returns:
        float or None: The lowest frequency (in Hz) of a peak with relative
                       prominence > 5. Returns None if no such peak is found.
    """
    if verbose:
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
        if verbose:
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
    strong_peak_mask = relative_prominences_arr > relative_prominence_threshold

    strong_peak_freqs = peak_frequencies[strong_peak_mask]
    strong_peak_psd_values = peak_psd_values[strong_peak_mask]
    strong_peak_std_proms = peak_prominences[strong_peak_mask]
    strong_peak_rel_proms = relative_prominences_arr[strong_peak_mask]

    if verbose:
        print(f"\n--- Peaks with Relative Prominence > {relative_prominence_threshold:.2f} ---")
    if len(strong_peak_freqs) > 0:
        # Sort by frequency for consistent output and selection of lowest
        sort_indices = np.argsort(strong_peak_freqs)
        strong_peak_freqs = strong_peak_freqs[sort_indices]
        strong_peak_psd_values = strong_peak_psd_values[sort_indices]
        strong_peak_std_proms = strong_peak_std_proms[sort_indices]
        strong_peak_rel_proms = strong_peak_rel_proms[sort_indices]

        if verbose:
            # Only print if it's the main call, not the rapid-fire tracking call
            if relative_prominence_threshold > 2.0:
                # Loop through the found strong peaks to print their details
                for i in range(len(strong_peak_freqs)):
                    print(f"  Peak at {strong_peak_freqs[i]:.2f} Hz, PSD: {strong_peak_psd_values[i]:.2e}, "
                          f"Std Prom: {strong_peak_std_proms[i]:.2f}, Rel Prom: {strong_peak_rel_proms[i]:.2f}")
        
        lowest_freq_strong_peak = strong_peak_freqs[0] # Already sorted by frequency
        if verbose:
            print(f"Returning lowest frequency with relative prominence > {relative_prominence_threshold:.2f}: {lowest_freq_strong_peak:.2f} Hz")
        return lowest_freq_strong_peak
    else:
        if verbose:
            print(f"No peaks found with relative prominence > {relative_prominence_threshold:.2f}.")
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

def remove_stim_artifacts_spline(raw_channel_data, pulse_starts_samples, pulse_ends_samples, sfreq, context_ms=10):
    """
    Removes stimulation artifacts from a channel's time series using spline interpolation.

    Args:
        raw_channel_data (np.ndarray): The 1D raw data for a single channel.
        pulse_starts_samples (list): List of start sample indices of detected pulses.
        pulse_ends_samples (list): List of end sample indices of detected pulses.
        sfreq (float): Sampling frequency of the data.
        context_ms (float): Duration in milliseconds of "good" data to use before and after
                            each artifact for interpolation context.

    Returns:
        np.ndarray: The signal with artifacts interpolated.
    """
    interpolated_signal = np.copy(raw_channel_data)
    n_samples = len(raw_channel_data)
    context_samples = int(context_ms / 1000 * sfreq)

    print(f"\n--- Performing spline interpolation to remove artifacts ({context_ms}ms context) ---")

    if not pulse_starts_samples:
        print("No pulses detected, returning original signal.")
        return interpolated_signal

    for i, (start, end) in enumerate(zip(pulse_starts_samples, pulse_ends_samples)):
        # Define the region to be interpolated
        interp_region_start = start
        interp_region_end = end

        # Define the context points for interpolation
        # Points before the artifact
        pre_context_indices = np.arange(max(0, interp_region_start - context_samples), interp_region_start)
        # Points after the artifact
        post_context_indices = np.arange(interp_region_end + 1, min(n_samples, interp_region_end + 1 + context_samples))

        x_known = np.concatenate((pre_context_indices, post_context_indices))
        y_known = raw_channel_data[x_known]

        # Determine interpolation kind based on number of known points
        interp_kind = 'linear'
        if len(x_known) >= 4: # Cubic requires at least 4 points
            interp_kind = 'cubic'
        elif len(x_known) >= 2: # Linear requires at least 2 points
            interp_kind = 'linear'
        else:
            print(f"Warning: Not enough context points ({len(x_known)}) for interpolation around artifact {i+1} (samples {start}-{end}). Skipping this artifact.")
            continue # Skip this artifact if not enough context

        # Create the interpolation function
        try:
            f_interp = scipy.interpolate.interp1d(x_known, y_known, kind=interp_kind, bounds_error=False, fill_value='extrapolate')
        except ValueError as e:
            print(f"Error creating interpolation function for artifact {i+1} (samples {start}-{end}): {e}. Skipping.")
            continue

        # Define the points within the artifact region to fill
        x_to_fill = np.arange(interp_region_start, interp_region_end + 1)
        
        # Perform interpolation
        interpolated_values = f_interp(x_to_fill)
        
        # Apply interpolated values to the signal copy
        interpolated_signal[x_to_fill] = interpolated_values
    
    print("Spline interpolation complete.")
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
    if stim_freq:
        ax1.axhline(stim_freq, color='red', linestyle='--', alpha=0.7, label=f'Stim Freq ({stim_freq:.1f} Hz)')
    ax1.legend(loc='upper right')
    fig.colorbar(pcm1, ax=ax1, label='Power/Frequency (dB/Hz)')

    # Calculate spectrogram for the cleaned signal, using the same color scale for fair comparison
    f_clean, t_clean, Sxx_clean = signal.spectrogram(cleaned_signal, fs=sfreq, nperseg=int(sfreq))
    pcm2 = ax2.pcolormesh(t_clean, f_clean, 10 * np.log10(Sxx_clean + np.finfo(float).eps), shading='gouraud', cmap='viridis', vmin=pcm1.get_clim()[0], vmax=pcm1.get_clim()[1])
    ax2.set_title('After Artifact Removal')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    if stim_freq:
        ax2.axhline(stim_freq, color='red', linestyle='--', alpha=0.7)
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

def calculate_frequency_over_time(all_channels_data, sfreq, window_sec=2.0, step_sec=0.5):
    """
    Calculates the dominant stimulation frequency in a signal over time using a sliding window.
    This version uses the full multi-step refinement process within each window.

    Args:
        all_channels_data (np.ndarray): The 2D raw data for ALL channels (n_channels, n_samples).
        sfreq (float): Sampling frequency.
        window_sec (float): Duration of the sliding window in seconds.
        step_sec (float): Step size for the sliding window in seconds.

    Returns:
        tuple: (window_times, frequencies_over_time) where frequencies_over_time
               may contain np.nan for windows where no peak was found.
    """
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    n_channels, n_samples = all_channels_data.shape

    window_starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    window_times = (window_starts + window_samples / 2) / sfreq # Center of window

    frequencies_over_time = []
    
    print(f"\nAnalyzing frequency over time with a {window_sec}s window, stepping every {step_sec}s (using full refinement)...")
    for i, start in enumerate(window_starts):
        print(f"\rProcessing window {i+1}/{len(window_starts)}...", end='', flush=True)
        end = start + window_samples
        window_data = all_channels_data[:, start:end]

        # 1. Calculate PSD for all channels in this window
        psds_win, freqs_win = psd_array_multitaper(
            window_data, 
            sfreq=sfreq, 
            fmin=1, 
            fmax=sfreq/2, 
            verbose=False
        )

        # 2. Get initial estimate from mean PSD of the window
        mean_psd_win = np.mean(psds_win, axis=0)
        est_freq_win = find_prominent_peaks(
            psd_values=mean_psd_win,
            frequencies=freqs_win,
            prominence=0.1,
            distance=20,
            relative_prominence_threshold=2.0, # Lenient for tracking
            verbose=False # Suppress printing for this call
        )

        final_freq_for_window = np.nan # Default to NaN

        # 3. Refine using the strongest channel in the window
        if est_freq_win is not None and est_freq_win > 0:
            band_total_width = 1.0 / (est_freq_win * 10.0)
            band_half_width = band_total_width / 2.0
            lower_b = max(est_freq_win - band_half_width, freqs_win.min())
            upper_b = min(est_freq_win + band_half_width, freqs_win.max())

            if upper_b > lower_b:
                freq_mask = (freqs_win >= lower_b) & (freqs_win <= upper_b)
                if np.any(freq_mask):
                    channel_band_powers = np.sum(psds_win[:, freq_mask], axis=1)
                    strong_channel_idx_win = np.argmax(channel_band_powers)
                    
                    refined_freq = find_prominent_peaks(
                        psd_values=psds_win[strong_channel_idx_win, :], frequencies=freqs_win,
                        prominence=0.1, distance=20, relative_prominence_threshold=2.0, verbose=False
                    )
                    final_freq_for_window = refined_freq if refined_freq is not None else est_freq_win
                else:
                    final_freq_for_window = est_freq_win # Fallback if band is empty
            else:
                final_freq_for_window = est_freq_win # Fallback if band is invalid
        
        frequencies_over_time.append(final_freq_for_window)
    
    print("\nFrequency tracking complete.")
    return window_times, np.array(frequencies_over_time)

def plot_frequency_over_time(times, frequencies, channel_name, mean_freq=None):
    """Plots the estimated stimulation frequency over time."""
    plt.figure(figsize=(18, 6))
    plt.plot(times, frequencies, marker='.', linestyle='-', label='Frequency per Window')
    
    if mean_freq is not None:
        plt.axhline(mean_freq, color='red', linestyle='--', label=f'Overall Estimated Freq: {mean_freq:.2f} Hz')
    
    valid_freqs = frequencies[~np.isnan(frequencies)]
    if len(valid_freqs) > 0:
        y_min, y_max = np.min(valid_freqs) - 5, np.max(valid_freqs) + 5
        plt.ylim(max(0, y_min), y_max)

    plt.title(f'Stimulation Frequency Over Time (Channel: {channel_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Estimated Frequency (Hz)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

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

        # --- New Analysis: Frequency over Time ---
        if strong_channel_idx_for_viz is not None and final_est_stim_freq is not None:
            print("\n--- Calculating Stimulation Frequency Over Time ---")
            
            # Call new function to calculate frequency over time
            window_times, freqs_over_time = calculate_frequency_over_time(
                all_channels_data=data, # Pass all channel data
                sfreq=sampleRate,
            )
            
            # Call new function to plot the results
            plot_frequency_over_time(
                times=window_times,
                frequencies=freqs_over_time,
                channel_name=raw.ch_names[strong_channel_idx_for_viz],
                mean_freq=final_est_stim_freq
            )

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
                    pulse_starts_samples=merged_starts,
                    pulse_ends_samples=merged_ends,
                    sfreq=sampleRate,
                    context_ms=10 # 10ms context on each side
                )

                # Step 5: Separate pulses by duration for plotting and then graph the results
                normal_pulses = []
                long_pulses = []
                if merged_starts:
                    pulse_durations = np.array(merged_ends) - np.array(merged_starts)
                    # Define the threshold for the top 1% longest pulses
                    duration_threshold = np.percentile(pulse_durations, 99)
                    print(f"Highlighting pulses longer than the 99th percentile duration ({duration_threshold:.0f} samples).")

                    for start, end, duration in zip(merged_starts, merged_ends, pulse_durations):
                        if duration >= duration_threshold:
                            long_pulses.append((start, end))
                        else:
                            normal_pulses.append((start, end))
                else: # No pulses, so both lists are empty
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