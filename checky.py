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

# --- New helper function for D3 burst detection ---
def detect_d3_bursts(third_derivative, sfreq, stim_freq_hz):
    """
    Detects high-frequency bursts in a 3rd derivative signal using Hilbert transform
    with hysteresis thresholding. Also identifies bursts that are very close to each other.

    A burst starts when the envelope crosses a high threshold and ends when it drops
    below a low threshold.

    Args:
        third_derivative (np.ndarray): The "pure" 3rd derivative of the signal.
        sfreq (float): Sampling frequency.
        stim_freq_hz (float): Estimated stimulation frequency.

    Returns:
        tuple: (hilbert_starts, hilbert_ends, amplitude_envelope, high_threshold, too_close_mask)
    """
    if len(third_derivative) == 0:
        return np.array([]), np.array([]), np.array([]), 0, np.array([])

    print(f"Analyzing 3rd derivative for bursts on pure signal (using Hilbert with Hysteresis)")

    # 1. Compute amplitude envelope on the raw (unsmoothed) 3rd derivative signal
    analytic_signal_d3 = signal.hilbert(third_derivative)
    amplitude_envelope_d3 = np.abs(analytic_signal_d3)
    
    # 2. Define separate upswing and downswing thresholds for burst detection.
    # A burst STARTS when the envelope crosses the upswing_threshold.
    # Then, it must cross a 'middle_threshold'.
    # A burst ENDS when the envelope subsequently drops below the downswing_threshold.
    
    # Calculate statistics from the envelope for the new thresholding logic
    median_env = np.median(amplitude_envelope_d3)
    percentile_98_env = np.percentile(amplitude_envelope_d3, 98)
    
    # Upswing threshold: median + 10% of the difference between the 98th percentile and the median
    upswing_threshold = median_env + 0.050 * (percentile_98_env - median_env)
    # Middle threshold: median + 50% of the difference between the 98th percentile and the median
    middle_threshold = median_env + 0.5 * (percentile_98_env - median_env)
    # Downswing threshold is now calculated dynamically for each burst.
    
    print(f"  Burst thresholds: Upswing={upswing_threshold:.2e}, Middle={middle_threshold:.2e}, Downswing=Dynamic (post-middle-threshold minimum)")
    
    # 3. Detect pulses using the three-threshold state machine logic
    hilbert_starts = []
    hilbert_ends = []
    
    # State machine flags
    potential_burst_start_idx = -1
    dynamic_downswing_threshold = -1.0 # Will be set for each burst
    in_potential_burst = False # True if crossed upswing_threshold
    in_actual_burst = False    # True if crossed middle_threshold (and thus upswing)

    for i, val in enumerate(amplitude_envelope_d3): # Iterate over raw amplitude_envelope_d3
        if not in_actual_burst: # Not currently in an active burst
            if not in_potential_burst: # Waiting for initial upswing
                if val > upswing_threshold:
                    potential_burst_start_idx = i
                    in_potential_burst = True
            else: # In potential burst, waiting for middle threshold
                if val > middle_threshold:
                    # Transition from potential to actual burst
                    hilbert_starts.append(potential_burst_start_idx)
                    in_actual_burst = True
                    in_potential_burst = False # Reset potential flag

                    # --- NEW DOWNSWING LOGIC ---
                    # Calculate the dynamic downswing threshold for THIS burst
                    if stim_freq_hz is not None and stim_freq_hz > 0:
                        window_duration_s = 1.0 / (2.0 * stim_freq_hz)
                        window_duration_samples = int(window_duration_s * sfreq)
                        search_start_idx = i # Start search from current point
                        search_end_idx = min(search_start_idx + window_duration_samples, len(amplitude_envelope_d3))

                        if search_end_idx > search_start_idx:
                            # The downswing threshold is the minimum value in this window
                            dynamic_downswing_threshold = np.min(amplitude_envelope_d3[search_start_idx:search_end_idx])
                        else:
                            # Fallback: if window is invalid, use the median as a safe bet
                            dynamic_downswing_threshold = median_env
                    else:
                        # Fallback: if no stim freq, use median
                        dynamic_downswing_threshold = median_env
                    # --- END NEW DOWNSWING LOGIC ---

                elif val < upswing_threshold: # Dropped below upswing before hitting middle
                    in_potential_burst = False # Reset potential flag
                    potential_burst_start_idx = -1
        else: # Currently in an actual burst
            if val < dynamic_downswing_threshold:
                hilbert_ends.append(i)
                in_actual_burst = False
                in_potential_burst = False # Ensure both are reset
                dynamic_downswing_threshold = -1.0 # Reset for next burst
    
    # If the signal ends while still in a burst, close the last burst
    if in_actual_burst:
        hilbert_ends.append(len(amplitude_envelope_d3) - 1) # Use length of raw envelope

    hilbert_starts = np.array(hilbert_starts)
    hilbert_ends = np.array(hilbert_ends)
    
    print(f"  Found {len(hilbert_starts)} raw bursts in D3 with hysteresis method.")

    d3_times = np.arange(len(third_derivative)) / sfreq

    # --- Step 1: Merge bursts that are very close together ("too close") ---
    merged_starts = []
    merged_ends = []
    if len(hilbert_starts) > 0 and stim_freq_hz is not None:
        merge_threshold_s = (1.0 / 10.0) / stim_freq_hz # Merge if gap is < 1/10th of a period
        
        merged_starts.append(hilbert_starts[0])
        merged_ends.append(hilbert_ends[0])

        for i in range(1, len(hilbert_starts)):
            # Gap is from the end of the last merged pulse to the start of the current pulse
            gap_s = d3_times[hilbert_starts[i]] - d3_times[merged_ends[-1]]

            if gap_s < merge_threshold_s:
                # Merge: update the end of the last merged pulse
                merged_ends[-1] = hilbert_ends[i]
            else:
                # No merge: add the current pulse as a new one
                merged_starts.append(hilbert_starts[i])
                merged_ends.append(hilbert_ends[i])
        
        if len(merged_starts) < len(hilbert_starts):
            print(f"  Merged {len(hilbert_starts) - len(merged_starts)} very close bursts. Final count: {len(merged_starts)}.")
    else:
        merged_starts = list(hilbert_starts)
        merged_ends = list(hilbert_ends)

    final_starts = np.array(merged_starts)
    final_ends = np.array(merged_ends)

    # --- Step 2: Highlight merged pulses that are "somewhat near" ---
    pulses_are_somewhat_near = np.zeros(len(final_starts), dtype=bool)
    if len(final_starts) > 1 and stim_freq_hz is not None:
        pulse_start_times_s = d3_times[final_starts]
        time_diffs_s = np.diff(pulse_start_times_s)
        
        # Highlight if distance between starts is less than half a stimulation period
        somewhat_near_threshold_s = 1.0 / (2.0 * stim_freq_hz)
        
        is_somewhat_near_mask = time_diffs_s < somewhat_near_threshold_s
        
        for i in range(len(is_somewhat_near_mask)):
            if is_somewhat_near_mask[i]:
                pulses_are_somewhat_near[i] = True
                pulses_are_somewhat_near[i+1] = True
        
        if np.any(pulses_are_somewhat_near):
            print(f"  Marked {np.sum(pulses_are_somewhat_near)} D3 bursts as being 'somewhat near' (closer than {somewhat_near_threshold_s*1000:.2f} ms).")

    return final_starts, final_ends, amplitude_envelope_d3, upswing_threshold, middle_threshold, pulses_are_somewhat_near

# --- New Plotting function for signal and its derivatives ---
def plot_signal_and_derivatives(
    signal_data,
    times_array,
    signal_name="Signal",
    stim_freq_hz=None,
    sfreq=None,
    d3_burst_starts_samples=None, # New arg for pre-computed D3 bursts
    d3_burst_ends_samples=None,   # New arg
    d3_amplitude_envelope=None,   # New arg
    d3_threshold=None,            # New arg
    d3_too_close_mask=None):      # New arg for highlighting
    """
    Plots the signal, its first derivative, and its second derivative.
    All subplots will have linked x-axes.

    Args:
        signal_data (np.ndarray): The 1D time series data.
        times_array (np.ndarray): The corresponding time vector for the signal.
        signal_name (str): Name of the signal for titles.
        stim_freq_hz (float, optional): Estimated stimulation frequency for title.
        sfreq (float, optional): Sampling frequency, required if pulse samples are provided.
        d3_burst_starts_samples (np.ndarray, optional): Pre-computed start samples of D3 bursts.
        d3_burst_ends_samples (np.ndarray, optional): Pre-computed end samples of D3 bursts.
        d3_amplitude_envelope (np.ndarray, optional): Pre-computed Hilbert envelope of D3.
        d3_threshold (float, optional): Pre-computed detection threshold for D3 bursts.
        d3_too_close_mask (np.ndarray, optional): Boolean mask for highlighting D3 bursts that are too close.
    """
    if len(signal_data) != len(times_array):
        print("Error in plot_signal_and_derivatives: signal_data and times_array must have the same length.")
        return

    # Calculate derivatives
    first_derivative = np.diff(signal_data)
    if len(first_derivative) > 0:
        second_derivative = np.diff(first_derivative)
        if len(second_derivative) > 0:
            third_derivative = np.diff(second_derivative)
            if len(third_derivative) > 0:
                fourth_derivative = np.diff(third_derivative)
            else: fourth_derivative = np.array([])
        else: third_derivative, fourth_derivative = np.array([]), np.array([])
    else:
        first_derivative, second_derivative, third_derivative, fourth_derivative = np.array([]), np.array([]), np.array([]), np.array([])
    fig, axes = plt.subplots(5, 1, figsize=(18, 20), sharex=True) # Original + 4 derivatives
    
    title_prefix = f'{signal_name} Analysis'
    if stim_freq_hz:
        title_prefix = f'{signal_name} Analysis (Est. Stim Freq: {stim_freq_hz:.2f} Hz)'
    fig.suptitle(title_prefix, fontsize=16)

    # Plot Original Signal
    axes[0].plot(times_array, signal_data, label='Original Signal', color='royalblue')
    axes[0].set_title('Original Time Series')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(loc='upper right')

    # Plot First Derivative
    # np.diff reduces length by 1, so times_array[1:] aligns with the derivative
    axes[1].plot(times_array[1:], first_derivative, label='First Derivative', color='forestgreen')
    axes[1].set_title('First Derivative')
    axes[1].set_ylabel('d(Amplitude)/dt')
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(loc='upper right') # Legend for the derivative line itself
    if len(first_derivative) > 0:
        max_abs_d1 = np.max(np.abs(first_derivative))
        if max_abs_d1 > 0: # Avoid setting ylim to [0,0] if derivative is flat zero
            axes[1].set_ylim(-max_abs_d1 * 1.1, max_abs_d1 * 1.1) # Add 10% padding


    # Plot Second Derivative
    # np.diff applied twice reduces length by 2
    axes[2].plot(times_array[2:], second_derivative, label='Second Derivative', color='darkorange')
    axes[2].set_title('Second Derivative')
    axes[2].set_ylabel('d^2(Amplitude)/dt^2')
    axes[2].grid(True, alpha=0.5)
    axes[2].legend(loc='upper right') # Legend for the derivative line itself
    if len(second_derivative) > 0:
        max_abs_d2 = np.max(np.abs(second_derivative))
        if max_abs_d2 > 0: # Avoid setting ylim to [0,0]
            axes[2].set_ylim(-max_abs_d2 * 1.1, max_abs_d2 * 1.1) # Add 10% padding

    # Plot Third Derivative
    axes[3].plot(times_array[3:], third_derivative, label='Third Derivative', color='purple')
    axes[3].set_title('Third Derivative')
    axes[3].set_ylabel('d^3(Amplitude)/dt^3')
    axes[3].grid(True, alpha=0.5)
    # Legend for D3 will be handled after potential burst plotting
    if len(third_derivative) > 0:
        max_abs_d3 = np.max(np.abs(third_derivative))
        if max_abs_d3 > 0:
            axes[3].set_ylim(-max_abs_d3 * 1.1, max_abs_d3 * 1.1)
    
    # Oscillation burst detection and highlighting in 3rd derivative for strong channel
    if "Average" not in signal_name and d3_burst_starts_samples is not None and \
       len(d3_burst_starts_samples) > 0 and len(third_derivative) > 0:
        
        d3_times = times_array[3:]
        
        # Plot the pre-computed envelope and threshold
        if d3_amplitude_envelope is not None and d3_threshold is not None:
            axes[3].plot(d3_times, d3_amplitude_envelope, color='cyan', linestyle='--', 
                         label='Hilbert Envelope (D3)', alpha=0.9)
            axes[3].axhline(d3_threshold, color='lime', linestyle=':', linewidth=2,
                            label=f'Middle Threshold (D3)')
        
        # Highlight the pre-computed bursts
        first_hilbert_burst_span = True
        first_error_burst_span = True
        for i, (start_idx, end_idx) in enumerate(zip(d3_burst_starts_samples, d3_burst_ends_samples)):
            if start_idx < len(d3_times) and end_idx < len(d3_times):
                start_time, end_time = d3_times[start_idx], d3_times[end_idx]
                
                # Check the 'too close' mask for this pulse
                if d3_too_close_mask is not None and d3_too_close_mask[i]:
                    label = "D3 Bursts (Too Close)" if first_error_burst_span else "_nolegend_"
                    axes[3].axvspan(start_time, end_time, color='magenta', alpha=0.5, label=label)
                    if first_error_burst_span: first_error_burst_span = False
                else:
                    label = "Hilbert-Detected Bursts" if first_hilbert_burst_span else "_nolegend_"
                    axes[3].axvspan(start_time, end_time, color='yellow', alpha=0.3, label=label)
                    if first_hilbert_burst_span: first_hilbert_burst_span = False
        
        # After all analysis on this axis, call legend to show all labels.
        axes[3].legend(loc='upper right')
    else: # If no burst analysis, ensure the default legend for the derivative line is there
        axes[3].legend(loc='upper right')

    # Plot Fourth Derivative
    axes[4].plot(times_array[4:], fourth_derivative, label='Fourth Derivative', color='brown')
    axes[4].set_title('Fourth Derivative')
    axes[4].set_ylabel('d^4(Amplitude)/dt^4')
    axes[4].set_xlabel('Time (s)') # X-label only on the last plot
    axes[4].legend(loc='upper right')
    if len(fourth_derivative) > 0:
        max_abs_d4 = np.max(np.abs(fourth_derivative))
        if max_abs_d4 > 0:
            axes[4].set_ylim(-max_abs_d4 * 1.1, max_abs_d4 * 1.1)
    axes[4].grid(True, alpha=0.5) # Grid after potential ylim adjustment


    # Disable y-axis autoscaling for derivative plots after initial setup.
    # This prevents y-panning while allowing y-zooming via specific zoom tools.
    # Apply to all derivative axes
    for i in range(1, 5): # axes[1] through axes[4]
        if len(axes[i].lines) > 0 : # Check if plot is not empty
             # Check if corresponding derivative data exists before trying to autoscale
            if i == 1 and len(first_derivative) == 0: continue
            if i == 2 and len(second_derivative) == 0: continue
            if i == 3 and len(third_derivative) == 0: continue
            if i == 4 and len(fourth_derivative) == 0: continue
            axes[i].autoscale(enable=False, axis='y')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    # plt.show() will be called once in main


# --- New Plotting function for Signal with Interpolated D3 Bursts ---
def plot_signal_with_interpolated_d3_bursts(
    signal_data_to_plot,
    times_array,
    sfreq,
    d3_burst_starts_samples, # These are indices into the D3 array
    d3_burst_ends_samples,   # These are indices into the D3 array
    channel_name="Signal"
):
    """
    Plots a signal, highlighting D3 Hilbert-detected bursts and showing the signal
    with cubic/linear spline interpolation applied within these burst regions.
    """
    if signal_data_to_plot is None or len(signal_data_to_plot) == 0 or \
       d3_burst_starts_samples is None or len(d3_burst_starts_samples) == 0:
        print("plot_signal_with_interpolated_d3_bursts: Insufficient data provided.")
        return

    signal_copy_interpolated = signal_data_to_plot.copy()
    d3_burst_times_for_plot = [] # To store (start_time, end_time) for axvspan

    # The burst samples are for the D3 array. We need to map them to the original signal array.
    # D3 is 3 samples shorter. So, an index `i` in D3 corresponds to index `i+3` in the original signal.
    for start_samp_d3, end_samp_d3 in zip(d3_burst_starts_samples, d3_burst_ends_samples):
        gap_start_samp = start_samp_d3 + 3
        gap_end_samp = end_samp_d3 + 3

        if gap_end_samp >= len(signal_data_to_plot) or gap_start_samp >= gap_end_samp:
            continue

        # Store times for axvspan
        start_time = times_array[gap_start_samp]
        end_time = times_array[gap_end_samp]
        d3_burst_times_for_plot.append((start_time, end_time))

        # --- Interpolation for this gap ---
        x_known_samples, y_known_values = [], []
        num_pts_each_side = 2 # Number of points to try to get from each side of the gap

        for i in range(num_pts_each_side, 0, -1): # Points before
            idx = gap_start_samp - i
            if idx >= 0:
                x_known_samples.append(idx)
                y_known_values.append(signal_data_to_plot[idx])
        
        for i in range(1, num_pts_each_side + 1): # Points after
            idx = gap_end_samp + i
            if idx < len(signal_data_to_plot):
                x_known_samples.append(idx)
                y_known_values.append(signal_data_to_plot[idx])
        
        unique_x_indices = np.unique(x_known_samples, return_index=True)[1]
        x_known_samples = np.array(x_known_samples)[unique_x_indices]
        y_known_values = np.array(y_known_values)[unique_x_indices]
        
        sort_order = np.argsort(x_known_samples)
        x_known_samples, y_known_values = x_known_samples[sort_order], y_known_values[sort_order]

        samples_to_interpolate = np.arange(gap_start_samp, gap_end_samp + 1)
        if len(samples_to_interpolate) == 0: continue

        interp_kind = 'linear' if len(x_known_samples) < 4 else 'cubic'
        if len(x_known_samples) < 2: 
            print(f"Warning: Not enough points ({len(x_known_samples)}) for any interpolation at D3 burst starting {start_time:.3f}s. Skipping.")
            continue
        
        interp_func = scipy.interpolate.interp1d(x_known_samples, y_known_values, kind=interp_kind, bounds_error=False, fill_value="extrapolate")
        interpolated_segment = interp_func(samples_to_interpolate)
        signal_copy_interpolated[samples_to_interpolate] = interpolated_segment

    # Plotting
    plt.figure(figsize=(18, 7))
    plt.plot(times_array, signal_data_to_plot, label=f'Original {channel_name}', color='gray', alpha=0.4, zorder=1)
    plt.plot(times_array, signal_copy_interpolated, label=f'Interpolated {channel_name}', color='cornflowerblue', zorder=2)

    first_gap_span_plot = True
    for start_t, end_t in d3_burst_times_for_plot:
        label = "D3 Burst Region (Interpolated)" if first_gap_span_plot else "_nolegend_"
        plt.axvspan(start_t, end_t, color='lightgreen', alpha=0.3, label=label, zorder=0)
        if first_gap_span_plot: first_gap_span_plot = False
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Time Series of {channel_name} with D3 Hilbert Bursts Interpolated')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

# --- New Plotting function for D3 Burst Detection Summary ---
def plot_d3_burst_detection_summary(
    third_derivative,
    d3_times,
    amplitude_envelope,
    upswing_threshold,
    middle_threshold,
    detected_starts_samples,
    detected_ends_samples,
    channel_name="Signal"
):
    """
    Creates a dedicated plot showing the 3rd derivative, its envelope, all detection
    thresholds, and the final detected bursts.
    """
    plt.figure(figsize=(18, 7))
    ax = plt.gca()

    # Plot 1: 3rd derivative and its envelope
    ax.plot(d3_times, third_derivative, color='purple', alpha=0.6, label='Third Derivative', zorder=1)
    ax.plot(d3_times, amplitude_envelope, color='cyan', linestyle='--', label='Hilbert Envelope', zorder=2)

    # Plot 2: Thresholds
    ax.axhline(upswing_threshold, color='orange', linestyle=':', linewidth=2, label=f'Upswing Threshold ({upswing_threshold:.2e})', zorder=3)
    ax.axhline(middle_threshold, color='lime', linestyle=':', linewidth=2, label=f'Middle Threshold ({middle_threshold:.2e})', zorder=3)

    # Plot 3: Detected Bursts
    first_burst = True
    for start_samp, end_samp in zip(detected_starts_samples, detected_ends_samples):
        if start_samp < len(d3_times) and end_samp < len(d3_times):
            start_time, end_time = d3_times[start_samp], d3_times[end_samp]
            label = "Detected Burst" if first_burst else "_nolegend_"
            ax.axvspan(start_time, end_time, color='yellow', alpha=0.4, label=label, zorder=0)
            first_burst = False

    ax.set_title(f'Third Derivative Burst Detection Summary for {channel_name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('d^3(Amplitude)/dt^3')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.5)
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

        # 3. Perform D3 Burst Analysis and Plotting
        if final_est_stim_freq is not None:
            # Determine the source signal for analysis (full length)
            if strong_channel_idx_for_viz is not None:
                signal_for_plot = data[strong_channel_idx_for_viz]
                ch_name_for_plot = raw.ch_names[strong_channel_idx_for_viz]
                print(f"\nUsing full-length data from strong channel ({ch_name_for_plot}) for analysis.")
            else:
                signal_for_plot = avg_signal_full
                ch_name_for_plot = "Average"
                print("\nNo strong channel identified; using full-length average signal for analysis.")
            
            if len(signal_for_plot) > 0:
                # --- D3 Burst Detection and Plotting Logic ---
                d3_burst_starts, d3_burst_ends, d3_envelope, upswing_thresh, middle_thresh, d3_too_close_mask = (None,)*6

                # Only run D3 analysis on the strong channel, not the average
                if "Average" not in ch_name_for_plot:
                    # 1. Calculate D3 of the signal that will be plotted
                    d1 = np.diff(signal_for_plot)
                    d2 = np.diff(d1) if len(d1) > 0 else np.array([])
                    third_derivative = np.diff(d2) if len(d2) > 0 else np.array([])

                    # 2. Detect bursts in D3
                    d3_burst_starts, d3_burst_ends, d3_envelope, upswing_thresh, middle_thresh, d3_too_close_mask = detect_d3_bursts(
                        third_derivative, sampleRate, final_est_stim_freq
                    )

                # 3. Call the derivative plotter, passing D3 burst info (which might be None for average signal)
                plot_signal_and_derivatives(
                    signal_data=signal_for_plot,
                    times_array=raw.times,
                    signal_name=ch_name_for_plot,
                    stim_freq_hz=final_est_stim_freq,
                    sfreq=sampleRate,
                    d3_burst_starts_samples=d3_burst_starts,
                    d3_burst_ends_samples=d3_burst_ends,
                    d3_amplitude_envelope=d3_envelope,
                    d3_threshold=middle_thresh, # Use middle_thresh for the main derivative plot
                    d3_too_close_mask=d3_too_close_mask
                )

                # 4. Call the new interpolation plotter if D3 bursts were found
                if d3_burst_starts is not None and len(d3_burst_starts) > 0:
                    plot_signal_with_interpolated_d3_bursts(
                        signal_data_to_plot=signal_for_plot,
                        times_array=raw.times,
                        sfreq=sampleRate,
                        d3_burst_starts_samples=d3_burst_starts,
                        d3_burst_ends_samples=d3_burst_ends,
                        channel_name=ch_name_for_plot
                        )

                    # 5. Call the new D3 summary plotter
                    d3_times_vec = raw.times[3:] # Time vector for 3rd derivative
                    plot_d3_burst_detection_summary(
                        third_derivative=third_derivative,
                        d3_times=d3_times_vec[:len(third_derivative)], # Ensure lengths match
                        amplitude_envelope=d3_envelope,
                        upswing_threshold=upswing_thresh,
                        middle_threshold=middle_thresh,
                        detected_starts_samples=d3_burst_starts,
                        detected_ends_samples=d3_burst_ends,
                        channel_name=ch_name_for_plot
                    )

            else: # if len(signal_for_plot) == 0
                print("Signal for analysis is empty, skipping D3 burst detection and plotting.")
        else: # if not final_est_stim_freq
            print("No stimulation frequency estimated, skipping D3 burst detection and plotting.")

        # Show all generated Matplotlib figures at the end
        plt.show()

    except SystemExit:
        print("Program execution cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")