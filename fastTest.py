#change so that you can store/reuse data

import tkinter as tk
from tkinter import filedialog
import os

import mne
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # For more control over subplots
from mne.time_frequency import psd_array_multitaper
from scipy.signal import find_peaks

# --- Configuration Switch for Testing ---
USE_DEFAULT_TEST_PATHS = True # Set to False to use file dialogs

# --- Define Default Paths (EDIT THESE FOR YOUR SYSTEM) ---
if USE_DEFAULT_TEST_PATHS:
    DEFAULT_FIF_FILE_PATH = "/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw.fif" # Replace with your actual .fif path
    DEFAULT_NPZ_FILE_PATH = "/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw_psds.npz" # Replace with your actual .npz path

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
    fif_file_path = DEFAULT_FIF_FILE_PATH
    npz_file_path = DEFAULT_NPZ_FILE_PATH
    print(f"--- USING DEFAULT TEST PATHS ---")
    print(f"Default .fif file: {fif_file_path}")
    print(f"Default .npz file: {npz_file_path}")
    if not os.path.exists(fif_file_path):
        print(f"Error: Default .fif file not found at '{fif_file_path}'. Please check the path or set USE_DEFAULT_TEST_PATHS to False.")
        raise SystemExit
    if not os.path.exists(npz_file_path):
        print(f"Error: Default .npz file not found at '{npz_file_path}'. Please check the path or set USE_DEFAULT_TEST_PATHS to False.")
        raise SystemExit
else:
    # Step 1: Select the EEG .fif file (for metadata and raw data)
    fif_file_path = filedialog.askopenfilename(
        title="Select the EEG .fif file (for raw data and metadata)",
        filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
        )
    if not fif_file_path:
        print("No .fif file was selected. Exiting the program.")
        raise SystemExit

    # Step 2: Select the pre-calculated PSD .npz file
    npz_file_path = filedialog.askopenfilename(
        title="Select the pre-calculated PSD .npz file",
        filetypes=[("NumPy archive files", "*.npz"), ("All files", "*.*")]
    )
    if not npz_file_path:
        print("No .npz file was selected. Exiting the program.")
        raise SystemExit

print(f"Loading .fif file: {fif_file_path}")
raw = mne.io.read_raw_fif(fif_file_path, preload=True)

print(f"Loading PSD data from: {npz_file_path}")

try:
    psd_data_loaded = np.load(npz_file_path)
    psds = psd_data_loaded['psds']
    freqs = psd_data_loaded['freqs']
    print("PSDs and frequencies loaded successfully from .npz file.")
    if psds.shape[0] != len(raw.ch_names):
        print(f"Warning: Mismatch in channel count between .npz file ({psds.shape[0]}) "
              f"and .fif file ({len(raw.ch_names)}). Ensure compatibility.")
    if freqs.ndim != 1:
        print(f"Warning: 'freqs' array from .npz is not 1-dimensional. This might cause issues.")
except KeyError as e:
    print(f"Error: The .npz file '{npz_file_path}' is missing a required key: {e}. Expected 'psds' and 'freqs'.")
    raise SystemExit
except Exception as e:
    print(f"Error loading .npz file '{npz_file_path}': {e}")
    raise SystemExit

# Parameters derived from the loaded .fif file

sampleRate = raw.info['sfreq']
data = raw.get_data()                # shape (n_chan, n_samples)


n_channels = len(raw.ch_names)

n_chan_fif, _ = data.shape # Number of channels from .fif file
n_chan_psd = psds.shape[0]   # Number of channels from loaded PSDs

print(f"The .fif recording has {n_chan_fif} channels. Loaded PSDs have {n_chan_psd} channels.")
if n_chan_fif != n_chan_psd:
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






# --- 5. Visualize Stimulation Epoch with a Sliding Window ---
def visualize_stim_epoch_with_sliding_window(
    raw, 
    stim_freq_hz, 
    window_sec=1.0, 
    step_sec=0.1, 
    threshold_factor=1.5,
    grace_period_sec=5.0,
    strong_channel_idx=None
    ) -> None: # Added return type hint for clarity
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
        grace_period_sec (float): The maximum duration (in seconds) of sub-threshold
                                  activity allowed before an epoch is considered ended.
        strong_channel_idx (int, optional): Index of the channel to display in the
                                            timeseries plot. Defaults to None.
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
    if strong_channel_idx is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True, 
                                           gridspec_kw={'height_ratios': [1, 2, 1]})
        fig.suptitle(f'Stimulation Epoch Analysis (Freq: {stim_freq_hz:.2f} Hz, Chan: {raw.ch_names[strong_channel_idx]})', fontsize=16)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 2]})
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
    
    # Determine start and end of the stimulation epoch with grace period
    is_active_window = np.array(prominence_over_time) > detection_threshold
    active_window_indices = np.where(is_active_window)[0]

    epoch_start_time = None
    epoch_end_time = None

    if len(active_window_indices) > 0:
        epoch_start_time = window_times[active_window_indices[0]]
        # This index points to the window_times array for the last window considered part of the epoch
        current_epoch_last_active_window_idx_in_times = active_window_indices[0] 

        for k in range(len(active_window_indices) - 1):
            idx_current_active = active_window_indices[k]
            idx_next_active = active_window_indices[k+1]
            
            time_diff_starts = window_times[idx_next_active] - window_times[idx_current_active]
            
            if time_diff_starts <= (grace_period_sec + window_sec):
                current_epoch_last_active_window_idx_in_times = idx_next_active
            else:
                break # Gap is too large, epoch ended with window at idx_current_active
        epoch_end_time = window_times[current_epoch_last_active_window_idx_in_times] + window_sec
        
        # Highlight on spectrogram
        ax2.axvspan(epoch_start_time, epoch_end_time, color='orangered', alpha=0.3, 
                    label=f'Detected Epoch ({epoch_start_time:.2f}s - {epoch_end_time:.2f}s)')
    
    # Plot 3: Timeseries of the strong channel if provided
    if strong_channel_idx is not None and 'ax3' in locals():
        channel_data_strong = raw.get_data(picks=[strong_channel_idx])[0]
        times_raw = raw.times
        ax3.plot(times_raw, channel_data_strong, color='purple', alpha=0.7, label=f'Channel: {raw.ch_names[strong_channel_idx]}')
        if epoch_start_time is not None and epoch_end_time is not None:
            ax3.axvspan(epoch_start_time, epoch_end_time, color='orangered', alpha=0.3)
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('Time (s)')
        ax3.set_title(f'Timeseries of Strong Channel ({raw.ch_names[strong_channel_idx]}) with Detected Epoch')
        ax3.legend(loc='upper right')
    ax2.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Helper function to get epoch times without immediate plotting
def determine_stim_epoch_boundaries(
    raw_obj,
    stim_freq_hz,
    window_sec=1.0,
    step_sec=0.1,
    threshold_factor=1.5,
    grace_period_sec=5.0
    ):
    """
    Determines the start and end times of the stimulation epoch.

    Args:
        raw_obj (mne.io.Raw): The loaded MNE Raw object.
        stim_freq_hz (float): The stimulation frequency to track.
        window_sec (float): Duration of the sliding window in seconds.
        step_sec (float): Step size for the sliding window in seconds.
        threshold_factor (float): Multiplier for median relative prominence for threshold.
        grace_period_sec (float): Max duration of sub-threshold activity to bridge epochs.

    Returns:
        tuple: (epoch_start_time_s, epoch_end_time_s) or (None, None) if no epoch found.
    """
    print("\n--- Determining stimulation epoch boundaries ---")
    data_full = raw_obj.get_data()
    sfreq_val = raw_obj.info['sfreq']
    _, n_samples_full = data_full.shape

    window_samples_val = int(window_sec * sfreq_val)
    step_samples_val = int(step_sec * sfreq_val)

    window_starts_val = np.arange(0, n_samples_full - window_samples_val + 1, step_samples_val) # Ensure last window fits
    if len(window_starts_val) == 0:
        print("Warning: Recording too short for the specified window and step size.")
        return None, None
        
    prominence_over_time_val = []

    for start_idx in window_starts_val:
        end_idx = start_idx + window_samples_val
        window_data_val = data_full[:, start_idx:end_idx]
        # Calculate PSD for this short window
        psds_win, freqs_win = psd_array_multitaper(window_data_val, sfreq=sfreq_val, fmin=1, fmax=sfreq_val/2, verbose=False)
        mean_psd_win = np.mean(psds_win, axis=0)
        # Find the frequency bin closest to our stimulation frequency
        stim_freq_idx_win = np.argmin(np.abs(freqs_win - stim_freq_hz))
        peak_power_win = mean_psd_win[stim_freq_idx_win]
        # Define neighborhood for relative prominence calculation
        neighborhood_hz_val = 5.0 # Consistent with visualize_stim_epoch_with_sliding_window
        lower_b = freqs_win[stim_freq_idx_win] - neighborhood_hz_val
        upper_b = freqs_win[stim_freq_idx_win] + neighborhood_hz_val
        neighborhood_mask_val = (freqs_win >= lower_b) & (freqs_win <= upper_b) & (freqs_win != freqs_win[stim_freq_idx_win])
        mean_neighborhood_power_val = np.mean(mean_psd_win[neighborhood_mask_val]) if np.any(neighborhood_mask_val) else 1e-10
        relative_prominence_val = peak_power_win / mean_neighborhood_power_val
        prominence_over_time_val.append(relative_prominence_val)

    # Use the same epoch determination logic as in visualize_stim_epoch_with_sliding_window
    detection_threshold_val = np.median(prominence_over_time_val) * threshold_factor
    is_active_window_val = np.array(prominence_over_time_val) > detection_threshold_val
    active_window_indices_val = np.where(is_active_window_val)[0]
    epoch_start_time_s, epoch_end_time_s = None, None # Initialize
    window_times_val = window_starts_val / sfreq_val

    if len(active_window_indices_val) > 0:
        epoch_start_time_s = window_times_val[active_window_indices_val[0]]
        current_epoch_last_active_window_idx_in_times = active_window_indices_val[0]
        for k_idx in range(len(active_window_indices_val) - 1):
            idx_current_active = active_window_indices_val[k_idx]
            idx_next_active = active_window_indices_val[k_idx+1]
            time_diff_starts = window_times_val[idx_next_active] - window_times_val[idx_current_active]
            if time_diff_starts <= (grace_period_sec + window_sec): # Check against combined duration
                current_epoch_last_active_window_idx_in_times = idx_next_active
            else:
                break 
        epoch_end_time_s = window_times_val[current_epoch_last_active_window_idx_in_times] + window_sec
        print(f"Determined epoch: {epoch_start_time_s:.2f}s - {epoch_end_time_s:.2f}s")
    else:
        print("Could not determine a distinct stimulation epoch. Template matching will use the full signal.")
    return epoch_start_time_s, epoch_end_time_s



# --- 6. Fully Automatic Pulse Identification Function ---
def find_stimulation_pulses_auto(
    avg_signal_epoch,
    sfreq,
    stim_freq_hz,
    debug_plots=False 
):
    # total template length is 1 / (frac_of_train_interval * stim_freq_hz)
    frac_of_train_interval = 4
    template_half_width_s = 1.0 / (2.0 * stim_freq_hz * frac_of_train_interval) 

    """
    Automatically detects stimulation pulses with varying numbers of components
    without requiring manual parameter tuning for the template shape.

    It works by:
    1. Finding a single "spiky" seed peak at the epoch onset.
    2. Extracting a template from a fixed-width window around this
       most prominent seed peak.
    3. Creating a template from these dynamic boundaries.
    4. Using this template in a matched filter to find all other pulses.

    Args:
        avg_signal_epoch (np.ndarray): The signal segment of the stimulation epoch.
        sfreq (float): Sampling frequency.
        stim_freq_hz (float): Estimated stimulation frequency.
        debug_plots (bool): If True, will generate a plot showing intermediate
                            steps of the automatic detection process.        
        (Note: template_half_width_s is now calculated internally based on stim_freq_hz
         to achieve a total template length of 1/(4*stim_freq_hz) seconds)

    Returns:
        tuple: (pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output, seed_peak_location_relative_to_epoch)
    """
    print("\n--- Running Fully Automatic Pulse Detection ---")

    # --- Step 1: Find a single "spiky" seed peak to anchor our analysis ---
    if stim_freq_hz <= 0:
        print("Automatic Strategy: stim_freq_hz must be positive.")
        return None, None, None, None, None
        
    # Define the search duration for the initial seed peak: +/- (1/2 * stim_freq_hz) around epoch start
    # This means the window width is 1 / (2 * stim_freq_hz)
    search_duration_s = 1.0 / (stim_freq_hz) ## change this if wanted
    search_end_sample = min(int(search_duration_s * sfreq), len(avg_signal_epoch))
    search_segment = avg_signal_epoch[:search_end_sample]

    if len(search_segment) < 3:
        print("Automatic Strategy: Epoch segment too short to find a seed peak.")
        return None, None, None, None, None

    # Find all candidate peaks in the absolute signal of the search_segment.
    # We will select the most prominent POSITIVE peak.
    # A minimal prominence might be useful to avoid picking up noise if the segment is flat,
    # We'll use a very small prominence value for find_peaks to get candidates,
    # and then select the one with the maximum calculated prominence.
    candidate_peaks_indices, properties = find_peaks(search_segment, prominence=1e-9) # Operate on original signal to find positive peaks
    
    # The main debug plot (if enabled) will show the seed peak selection in its first panel.
    if not candidate_peaks_indices.any():
        print(f"Automatic Strategy: No candidate POSITIVE seed peaks found in the initial search segment (length {len(search_segment)} samples).")
        return None, None, None, None, None

    # Filter out peaks at the very edges of the search_segment
    # A peak at index 0 or len(search_segment)-1 is an edge peak.
    # We require peaks to have at least one sample on either side within the search_segment.
    non_edge_mask = (candidate_peaks_indices > 0) & (candidate_peaks_indices < len(search_segment) - 1)
    candidate_peaks_indices_filtered = candidate_peaks_indices[non_edge_mask]
    candidate_prominences_filtered = properties["prominences"][non_edge_mask]

    if not candidate_peaks_indices_filtered.any():
        print(f"Automatic Strategy: All candidate POSITIVE peaks found were at the edges of the search segment ({len(search_segment)} samples long). No valid non-edge seed peak found.")
        # Optionally, you could plot the search_segment here if debug_plots is True to see why
        return None, None, None, None, None
        
    # Select the candidate peak that has the largest prominence value
    if not candidate_prominences_filtered.any(): # Should be caught by the check above
        print("Automatic Strategy: No prominences available for filtered candidate POSITIVE peaks.")
        return None, None, None, None, None

    idx_of_max_prominence_in_candidates = np.argmax(candidate_prominences_filtered)
    # Get the actual sample index in search_segment (and thus avg_signal_epoch)
    seed_peak_location = candidate_peaks_indices_filtered[idx_of_max_prominence_in_candidates]
    selected_peak_prominence = candidate_prominences_filtered[idx_of_max_prominence_in_candidates]
    
    print(f"Found POSITIVE seed peak (max prominence in initial {search_duration_s:.3f}s segment) at sample {seed_peak_location} "
          f"(prominence: {selected_peak_prominence:.2f}, value: {search_segment[seed_peak_location]:.2f}).")

    if debug_plots:
        plt.figure(figsize=(15, 8)) # Adjusted for 3 plots now
        
        # Plot 1: Seed peak finding
        ax_seed = plt.subplot(4, 1, 1)
        search_segment_times = np.arange(len(search_segment)) / sfreq
        ax_seed.plot(search_segment_times, search_segment, label='Initial Search Segment')
        # Plot all non-edge candidate peaks
        if candidate_peaks_indices_filtered.any():
            ax_seed.plot(search_segment_times[candidate_peaks_indices_filtered], search_segment[candidate_peaks_indices_filtered], 'o', label='Non-Edge Candidate Peaks', alpha=0.7)
        ax_seed.plot(search_segment_times[seed_peak_location], search_segment[seed_peak_location], 'rx', markersize=10, label='Chosen Seed Peak')
        ax_seed.set_title(f'Step 1: Seed Peak Detection (Found at {seed_peak_location/sfreq:.3f}s within segment)')
        ax_seed.legend()

    # --- Step 2: Extract Template based on seed_peak_location and fixed window ---
    samples_before_peak = int(template_half_width_s * sfreq)
    samples_after_peak = samples_before_peak # Symmetric window
    template_total_len = samples_before_peak + samples_after_peak

    if template_total_len <= 0:
        print(f"Automatic Strategy: Template total length is {template_total_len}. Check stim_freq_hz ({stim_freq_hz:.2f} Hz) and sfreq ({sfreq:.2f} Hz).")
        return None, None, None, None, seed_peak_location # Return seed_peak_location even on failure for potential debugging

    # Define window in avg_signal_epoch coordinates
    window_start_abs = seed_peak_location - samples_before_peak
    window_end_abs = seed_peak_location + samples_after_peak # Exclusive for slicing if used directly

    # Extract the snippet, padding with zeros at the edges
    template_waveform = np.zeros(template_total_len)
    
    # Source slice from avg_signal_epoch
    src_start_in_epoch = max(0, window_start_abs)
    src_end_in_epoch = min(len(avg_signal_epoch), window_end_abs)
    
    # Destination slice in template_waveform
    dest_start_in_template = max(0, -window_start_abs) # If window_start_abs is negative
    dest_end_in_template = dest_start_in_template + (src_end_in_epoch - src_start_in_epoch)

    if dest_start_in_template < dest_end_in_template and dest_end_in_template <= template_total_len:
        template_waveform[dest_start_in_template:dest_end_in_template] = avg_signal_epoch[src_start_in_epoch:src_end_in_epoch]
    
    # --- Scale the template waveform based on the seed peak amplitude ---
    # The seed_peak_location is an absolute index in avg_signal_epoch.
    # Its index within the extracted template_waveform is seed_peak_location - window_start_abs.
    seed_peak_index_in_template = seed_peak_location - window_start_abs

    # Ensure the index is valid within the extracted (potentially padded) template
    if 0 <= seed_peak_index_in_template < len(template_waveform):
        original_seed_amplitude = avg_signal_epoch[seed_peak_location]
        template_peak_amplitude_at_seed_loc = template_waveform[seed_peak_index_in_template]

        # Calculate scaling factor
        scaling_factor = 1.0 # Default to no scaling
        # Avoid division by zero or very small numbers
        if np.abs(template_peak_amplitude_at_seed_loc) > 1e-9:
             scaling_factor = original_seed_amplitude / template_peak_amplitude_at_seed_loc
             template_waveform = template_waveform * scaling_factor
             print(f"Scaled template waveform by factor: {scaling_factor:.4f}")
        else:
             print("Warning: Amplitude at seed peak location in template is near zero. Skipping template scaling.")
    else:
        print(f"Warning: Seed peak index ({seed_peak_index_in_template}) is outside template bounds ({len(template_waveform)}). Skipping template scaling.")

    print(f"Extracted template of length {len(template_waveform)} samples around seed peak (window: +/- {template_half_width_s * 1000:.2f} ms).")

    period_samples = int(sfreq / stim_freq_hz)
    
    peak_offset_in_template = 0 # Initialize
    if len(template_waveform) > 0:
        peak_offset_in_template = np.argmax(np.abs(template_waveform))

    if debug_plots:
        # Plot 3: Final Template Waveform (Dynamic Segment)
        ax_template_plot = plt.subplot(3, 1, 2) # Now subplot 2 of 3
        # Time axis for the dynamic template, centered
        dynamic_template_times = (np.arange(len(template_waveform)) - len(template_waveform)//2) / sfreq
        ax_template_plot.plot(dynamic_template_times, template_waveform, label=f'Final Template (Window: +/- {template_half_width_s * 1000:.2f} ms)')
        # Mark the prominent peak within the template
        time_of_peak_in_template_plot = (peak_offset_in_template - (len(template_waveform) // 2)) / sfreq
        ax_template_plot.axvline(time_of_peak_in_template_plot, color='lime', linestyle=':', linewidth=2, label=f'Prominent Peak in Template (New Effective Center)')
        ax_template_plot.set_title('Step 2: Final Template Waveform')
        ax_template_plot.legend()

    # --- Step 4: Matched filter and final detection ---
    mf_kernel = template_waveform[::-1]
    matched_filter_output = np.convolve(avg_signal_epoch, mf_kernel, mode='same')

    # MF threshold is removed as per request. We will find all peaks based on distance.
    mf_distance_samples = int(0.6 * period_samples)
    # This initial find_peaks call gives us all potential candidates, sorted by index.
    # The 'distance' parameter here helps to avoid picking multiple points from a single broad MF peak.
    all_mf_candidates, _ = find_peaks(
        matched_filter_output, height=None, distance=mf_distance_samples # height=None removes thresholding
    )

    if not all_mf_candidates.size > 0: # Check if array is not empty
        print("Automatic Strategy: No pulses found after matched filtering (initial pass).")
        return None, None, template_waveform, matched_filter_output, seed_peak_location

    # --- New logic: Refine the first pulse based on seed_peak_location and leeway ---
    print(f"Automatic Strategy: Initial detection found {len(all_mf_candidates)} pulses. Refining first pulse...")

    # Tolerance: +/- (3/4) of a stimulation period duration, centered on seed_peak_location
    final_detected_pulse_centers_list = []

    # --- First Pulse: Directly use the seed_peak_location ---
    if not (0 <= seed_peak_location < len(matched_filter_output)):
        print(f"Automatic Strategy: seed_peak_location ({seed_peak_location}) is out of bounds "
              f"for matched_filter_output (len: {len(matched_filter_output)}). Cannot set first pulse.")
        return None, None, template_waveform, matched_filter_output, seed_peak_location

    first_pulse_center = seed_peak_location
    final_detected_pulse_centers_list.append(first_pulse_center)
    print(f"Automatic Strategy: Set first pulse center to seed_peak_location: {first_pulse_center} "
          f"(MF score at seed: {matched_filter_output[first_pulse_center]:.2f})")
    last_confirmed_pulse_center = first_pulse_center

    # --- Iteratively find subsequent pulses ---
    # period_samples is already int(sfreq / stim_freq_hz)
    tolerance_samples = int(round(period_samples /4.0)) # +/- 1/4th of a period

    while True:
        # --- Define search window for the next pulse ---
        expected_next_center_ideal = last_confirmed_pulse_center + period_samples
        
        # Define search window
        search_window_min = expected_next_center_ideal - tolerance_samples
        search_window_max = expected_next_center_ideal + tolerance_samples

        # Ensure search window is within signal bounds for matched_filter_output
        # (also applies to avg_signal_epoch as they have the same length)
        if search_window_min >= len(avg_signal_epoch):
            print(f"Automatic Strategy: Search window for next pulse (min: {search_window_min}) is beyond signal length. Stopping search.")
            break
        
        # Clip window to be within bounds of matched_filter_output
        search_window_min_abs = max(0, search_window_min)
        search_window_max_abs = min(len(avg_signal_epoch) - 1, search_window_max)

        # --- Find the most prominent peak in avg_signal_epoch within this search window ---
        current_search_sub_epoch = avg_signal_epoch[search_window_min_abs : search_window_max_abs + 1]
        
        if len(current_search_sub_epoch) < 3: # find_peaks needs at least 3 samples for non-edge peaks
            print(f"Automatic Strategy: Search window segment (length {len(current_search_sub_epoch)}) is too short. Stopping search.")
            break 

        # Find positive-going peaks and select the one with max prominence in this window
        local_peak_indices_in_sub_epoch, local_properties = find_peaks(current_search_sub_epoch, prominence=1e-9) # Small prominence to get candidates

        if not local_peak_indices_in_sub_epoch.any():
            print(f"Automatic Strategy: No initial peaks found in avg_signal_epoch within window "
                  f"[{search_window_min_abs}, {search_window_max_abs}] around expected center {expected_next_center_ideal:.0f}. Stopping search.")
            break
        
        # Filter out peaks at the very edges of the current_search_sub_epoch for robust prominence.
        non_edge_mask_local = (local_peak_indices_in_sub_epoch > 0) & \
                              (local_peak_indices_in_sub_epoch < len(current_search_sub_epoch) - 1)
        
        local_peak_indices_filtered = local_peak_indices_in_sub_epoch[non_edge_mask_local]
        
        if not local_peak_indices_filtered.any():
            print(f"Automatic Strategy: All peaks in window [{search_window_min_abs}, {search_window_max_abs}] "
                  f"were at the edges of the sub-segment. Cannot determine most prominent. Stopping search.")
            break

        local_prominences_filtered = local_properties["prominences"][non_edge_mask_local]
        if not local_prominences_filtered.any(): # Should be caught by above check
             print(f"Automatic Strategy: No prominences for non-edge peaks in window. Stopping search.")
             break

        idx_of_max_prom_in_local_peaks = np.argmax(local_prominences_filtered)
        best_local_peak_offset = local_peak_indices_filtered[idx_of_max_prom_in_local_peaks]
        
        # Convert local index back to absolute index in avg_signal_epoch
        next_pulse_center = search_window_min_abs + best_local_peak_offset
        
        # Safety check: ensure we are moving forward
        if next_pulse_center <= last_confirmed_pulse_center:
            print(f"Automatic Strategy: Newly found peak ({next_pulse_center}) is not after last confirmed peak ({last_confirmed_pulse_center}). Stopping.")
            break
        
        final_detected_pulse_centers_list.append(next_pulse_center)
        print(f"Automatic Strategy: Found next pulse at {next_pulse_center} (expected near {expected_next_center_ideal:.0f}, "
              f"tolerance: +/- {tolerance_samples} samples, raw signal prominence: {local_prominences_filtered[idx_of_max_prom_in_local_peaks]:.2f})")
        last_confirmed_pulse_center = next_pulse_center
        
    detected_pulse_centers = np.array(final_detected_pulse_centers_list)

    # --- Calculate pulse start and end based on the (potentially refined) detected_pulse_centers ---
    # For aligning the pulse window, we use the geometric center of the template,
    # ensuring that detected_pulse_centers corresponds to the center of the highlighted artifact.
    alignment_offset_in_template = len(template_waveform) // 2
            
    pulse_starts_samples = detected_pulse_centers - alignment_offset_in_template
    # Ensure pulse_ends_samples correctly reflects the full length of template_waveform.
    pulse_ends_samples = pulse_starts_samples + len(template_waveform)


    if debug_plots:
        # Plot 4: Matched Filter Output
        ax_mf = plt.subplot(3, 1, 3) # Now subplot 3 of 3
        epoch_times = np.arange(len(avg_signal_epoch)) / sfreq
        ax_mf.plot(epoch_times, matched_filter_output, label='Matched Filter Output')
        if detected_pulse_centers.any():
            ax_mf.plot(epoch_times[detected_pulse_centers], matched_filter_output[detected_pulse_centers], 'rx', markersize=8, label='Final Detected Pulse Centers')
        # ax_mf.axhline(mf_peak_threshold, color='k', linestyle='--', label=f'MF Threshold ({mf_peak_percentile}th percentile)') # Threshold line removed
        ax_mf.set_title('Step 4: Matched Filter Output and Detected Pulses')
        ax_mf.legend()
        plt.tight_layout() # Apply layout
        # plt.show() # Removed: will be called once in main

    print(f"Identified {len(pulse_starts_samples)} stimulation pulses automatically.")
    return pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output, seed_peak_location


# --- Plotting function for pulse identification results ---
def plot_template_and_overlaid_pulses(
    signal_to_plot,       # The time-domain signal (e.g., one channel or average)
    times_array,          # Time vector for signal_to_plot
    pulse_starts_samples, # Absolute start samples of pulses
    pulse_ends_samples,   # Absolute end samples of pulses    
    template_waveform,    # The template used for matching
    sfreq,                # Sampling frequency
    mean_signal_overall=None, # Default argument moved later
    channel_name="Signal",# Name of the signal being plotted
    stim_freq_hz=None,    # Estimated stimulation frequency (for title)
    seed_peak_abs_sample=None # Absolute sample index of the seed peak
):
    """
    Visualizes the derived template and the detected pulses overlaid on the time series.
    Panel 1: The derived template waveform.
    Panel 2: The signal with detected pulse durations highlighted.
    """
    fig, (ax_template, ax_signal) = plt.subplots(2, 1, figsize=(18, 10), sharex=False,
                                               gridspec_kw={'height_ratios': [1, 2]})

    title_str = f'Pulse Identification Results (Signal: {channel_name})'
    if stim_freq_hz:
        title_str = f'Pulse Identification Results (Est. Stim Freq: {stim_freq_hz:.2f} Hz, Signal: {channel_name})'
    fig.suptitle(title_str, fontsize=16)

    # panel 1 : template waveform
    if template_waveform is not None and len(template_waveform) > 0:
        template_time_axis = (np.arange(len(template_waveform)) - len(template_waveform) // 2) / sfreq # Centered
        ax_template.plot(template_time_axis, template_waveform, color='darkorange', label='Derived Template Waveform')
        ax_template.set_xlabel('Time relative to template center (s)')
        ax_template.set_ylabel('Amplitude')
        ax_template.set_title('Derived Template Waveform')
        ax_template.legend(loc='upper right')
    else:
        ax_template.text(0.5, 0.5, "Template not available or empty",
                         ha='center', va='center', transform=ax_template.transAxes, fontsize=12, color='grey')
        ax_template.set_title('Derived Template Waveform (Not Available)')
        ax_template.set_xlabel('Time relative to template center (s)')
        ax_template.set_ylabel('Amplitude')
    ax_template.grid(True, alpha=0.5)

    # Panel 2: Signal with Overlaid Detected Pulses (using the template)
    ax_signal.plot(times_array, signal_to_plot, label=f'Original Signal: {channel_name}', color='cornflowerblue', alpha=0.7)

    if mean_signal_overall is not None and channel_name != "Average" and len(mean_signal_overall) == len(times_array):
        ax_signal.plot(times_array, mean_signal_overall, label='Overall Mean Signal', color='grey', linestyle=':', alpha=0.6)

    
    if pulse_starts_samples is not None and len(pulse_starts_samples) > 0 and \
       pulse_ends_samples is not None and len(pulse_starts_samples) == len(pulse_ends_samples):
        # Ensure pulse_ends_samples is also valid

        # Create a sorted list of all unique event points (starts and ends of pulses)
        all_event_points = np.unique(np.concatenate((pulse_starts_samples, pulse_ends_samples)))
        all_event_points.sort()

        # Flags for legend entries
        first_green_span = True
        first_red_span = True

        if len(all_event_points) > 1:
            for i in range(len(all_event_points) - 1):
                interval_start_sample = all_event_points[i]
                interval_end_sample = all_event_points[i+1]

                if interval_start_sample >= interval_end_sample: # Skip zero-duration or invalid intervals
                    continue

                # Determine how many original pulses cover the midpoint of this elementary interval
                mid_point_sample = (interval_start_sample + interval_end_sample) / 2.0
                overlap_count = 0
                for k in range(len(pulse_starts_samples)):
                    # A pulse k covers the midpoint if:
                    # pulse_starts_samples[k] <= mid_point_sample < pulse_ends_samples[k]
                    # Note: pulse_ends_samples is exclusive end for slicing, so use <
                    if pulse_starts_samples[k] <= mid_point_sample and mid_point_sample < pulse_ends_samples[k]:
                        overlap_count += 1
                
                # Convert interval samples to time for plotting
                start_time = interval_start_sample / sfreq
                end_time = interval_end_sample / sfreq

                if overlap_count == 1:
                    label = 'Detected Pulse Artifacts (No Overlap)' if first_green_span else '_nolegend_'
                    ax_signal.axvspan(start_time, end_time, color='green', alpha=0.4, label=label)
                    if first_green_span: first_green_span = False
                elif overlap_count > 1:
                    label = 'Overlapping Detected Artifacts' if first_red_span else '_nolegend_'
                    ax_signal.axvspan(start_time, end_time, color='red', alpha=0.5, label=label)
                    if first_red_span: first_red_span = False
        else: # Fallback for very few event points (e.g., single pulse, no distinct intervals)
            # This case might indicate an issue or a very simple scenario.
            # For simplicity, plot all as green if this fallback is hit.
            for i, (start_samp, end_samp) in enumerate(zip(pulse_starts_samples, pulse_ends_samples)):
                start_time = start_samp / sfreq
                end_time = end_samp / sfreq
                ax_signal.axvspan(start_time, end_time, color='green', alpha=0.4, label='Detected Pulse Artifacts' if i == 0 else '_nolegend_')

    # Plot the seed peak location if provided
    if seed_peak_abs_sample is not None:
        seed_peak_time = seed_peak_abs_sample / sfreq
        ax_signal.plot(seed_peak_time, signal_to_plot[seed_peak_abs_sample], 'm*', markersize=12, label=f'Initial Seed Peak ({seed_peak_time:.3f}s)')

    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude')
    ax_signal.set_title(f'Time Series of {channel_name} with Highlighted Pulse Artifacts')
    ax_signal.legend(loc='upper right')
    ax_signal.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle and bottom labels
    # plt.show() # Removed: will be called once in main

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

# --- New Plotting function for signal and its derivatives ---
def plot_signal_and_derivatives(
    signal_data,
    times_array,
    signal_name="Signal",
    stim_freq_hz=None,
    pulse_starts_samples=None,
    pulse_ends_samples=None,
    sfreq=None
):
    """
    Plots the signal, its first derivative, and its second derivative.
    All subplots will have linked x-axes.

    Args:
        signal_data (np.ndarray): The 1D time series data.
        times_array (np.ndarray): The corresponding time vector for the signal.
        signal_name (str): Name of the signal for titles.
        stim_freq_hz (float, optional): Estimated stimulation frequency for title.
        pulse_starts_samples (np.ndarray, optional): Array of start samples for each detected pulse.
        pulse_ends_samples (np.ndarray, optional): Array of end samples for each detected pulse.
        sfreq (float, optional): Sampling frequency, required if pulse samples are provided.
    """
    if len(signal_data) != len(times_array):
        print("Error in plot_signal_and_derivatives: signal_data and times_array must have the same length.")
        return

    # Calculate derivatives
    first_derivative = np.diff(signal_data)
    second_derivative = np.diff(first_derivative)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    
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
    if pulse_starts_samples is not None and pulse_ends_samples is not None and \
       len(pulse_starts_samples) > 0 and sfreq is not None:
        first_span = True
        first_max_deriv_marker = True # For legend of max derivative points
        for start_samp, end_samp in zip(pulse_starts_samples, pulse_ends_samples):
            start_time = start_samp / sfreq
            end_time = end_samp / sfreq
            label = 'Detected Pulses' if first_span else '_nolegend_'
            axes[0].axvspan(start_time, end_time, color='lightcoral', alpha=0.3, label=label)
            if first_span:
                first_span = False

            # Mark point of highest first derivative within this pulse on the original signal
            if end_samp > start_samp + 1: # Need at least 2 points for diff
                pulse_segment_signal = signal_data[start_samp:end_samp]
                if len(pulse_segment_signal) > 1: # Ensure segment is not too short
                    pulse_first_derivative = np.diff(pulse_segment_signal)
                    if len(pulse_first_derivative) > 0:
                        idx_max_deriv_local = np.argmax(pulse_first_derivative)
                        # The point on original signal corresponds to the end of the interval with max slope
                        # or idx_max_deriv_local if we consider the start. Let's use idx_max_deriv_local + 1.
                        abs_idx_max_deriv_pt = start_samp + idx_max_deriv_local + 1
                        
                        if abs_idx_max_deriv_pt < len(times_array): # Boundary check
                            marker_label = 'Max 1st Deriv in Pulse' if first_max_deriv_marker else '_nolegend_'
                            axes[0].plot(times_array[abs_idx_max_deriv_pt], signal_data[abs_idx_max_deriv_pt], 'X', color='magenta', markersize=8, label=marker_label)
                            if first_max_deriv_marker: first_max_deriv_marker = False
        axes[0].legend(loc='upper right') # Re-call legend

    # Plot First Derivative
    # np.diff reduces length by 1, so times_array[1:] aligns with the derivative
    axes[1].plot(times_array[1:], first_derivative, label='First Derivative', color='forestgreen')
    axes[1].set_title('First Derivative')
    axes[1].set_ylabel('d(Amplitude)/dt')
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(loc='upper right')
    if pulse_starts_samples is not None and pulse_ends_samples is not None and \
       len(pulse_starts_samples) > 0 and sfreq is not None:
        for start_samp, end_samp in zip(pulse_starts_samples, pulse_ends_samples):
            start_time = start_samp / sfreq
            end_time = end_samp / sfreq
            axes[1].axvspan(start_time, end_time, color='lightcoral', alpha=0.3)

    # Plot Second Derivative
    # np.diff applied twice reduces length by 2
    axes[2].plot(times_array[2:], second_derivative, label='Second Derivative', color='darkorange')
    axes[2].set_title('Second Derivative')
    axes[2].set_ylabel('d^2(Amplitude)/dt^2')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.5)
    axes[2].legend(loc='upper right')
    if pulse_starts_samples is not None and pulse_ends_samples is not None and \
       len(pulse_starts_samples) > 0 and sfreq is not None:
        for start_samp, end_samp in zip(pulse_starts_samples, pulse_ends_samples):
            start_time = start_samp / sfreq
            end_time = end_samp / sfreq
            axes[2].axvspan(start_time, end_time, color='lightcoral', alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    # plt.show() will be called once in main

# --- Main Execution Block (Corrected) ---
if __name__ == '__main__':
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

        # Note: plot_signal_and_derivatives will be called later, after pulse detection
        # 3. Determine and Visualize the Stimulation Epoch
        epoch_start_s, epoch_end_s = None, None
        if final_est_stim_freq is not None:
            epoch_start_s, epoch_end_s = determine_stim_epoch_boundaries(
                raw, final_est_stim_freq, threshold_factor=1.5
            )
            # You can uncomment the line below if you want to see the epoch visualization plot
            # visualize_stim_epoch_with_sliding_window(
            #     raw, final_est_stim_freq, strong_channel_idx=strong_channel_idx_for_viz
            # )
        else:
            print("\nNo suitable stimulation frequency found to visualize epoch.")

        # 4. Perform Pulse Identification using the FULLY AUTOMATIC function
        if final_est_stim_freq:
            # Determine the source signal for template creation
            if strong_channel_idx_for_viz is not None:
                source_signal_for_template_creation = data[strong_channel_idx_for_viz]
                print(f"\nUsing data from strong channel ({raw.ch_names[strong_channel_idx_for_viz]}) for template creation.")
            else:
                source_signal_for_template_creation = avg_signal_full
                print("\nNo strong channel identified or specified; using average signal for template creation.")

            # Epoch the source signal if epoch boundaries are defined
            if epoch_start_s is not None and epoch_end_s is not None:
                signal_for_template = source_signal_for_template_creation[int(epoch_start_s*sampleRate):int(epoch_end_s*sampleRate)]
                offset_samples = int(epoch_start_s*sampleRate)
            else: # Use the full signal
                signal_for_template = source_signal_for_template_creation
                offset_samples = 0
            
            if len(signal_for_template) > 0:
                # Call the final, automatic function.
                pulse_starts_rel, pulse_ends_rel, template, mf_out, seed_peak_rel = find_stimulation_pulses_auto(
                    avg_signal_epoch=signal_for_template,
                    sfreq=sampleRate,
                    stim_freq_hz=final_est_stim_freq,
                    debug_plots=USE_DEFAULT_TEST_PATHS # Automatically enable debug plots if using default paths
                )

                # 5. Plot the Final Results
                # Initialize plot variables to safe defaults
                pulse_starts_abs, pulse_ends_abs = None, None # Add pulse_ends_abs back
                current_template_for_plot = np.array([]) # Ensure it's an empty array, not None
                seed_peak_abs_for_plot = None

                if pulse_starts_rel is not None and len(pulse_starts_rel) > 0:
                    pulse_starts_abs = np.array(pulse_starts_rel) + offset_samples
                    pulse_ends_abs = np.array(pulse_ends_rel) + offset_samples
                    if seed_peak_rel is not None: seed_peak_abs_for_plot = seed_peak_rel + offset_samples

                    signal_for_plot = data[strong_channel_idx_for_viz] if strong_channel_idx_for_viz is not None else avg_signal_full
                    ch_name_for_plot = raw.ch_names[strong_channel_idx_for_viz] if strong_channel_idx_for_viz is not None else "Average"

                    if template is not None and len(template) > 0:
                        current_template_for_plot = np.asarray(template).squeeze()
                        if current_template_for_plot.ndim == 0 or len(current_template_for_plot) == 0:
                            print("Warning: Template is scalar or empty after squeeze. MF output will be zero for plotting.")
                            current_template_for_plot = np.array([]) # Ensure it's an empty array for plotting func
                        # else:
                            # mf_full_for_plot and detected_pulse_centers_abs_for_plot were calculated here but are not used by the current plot
                    else:
                        print("Template not available or empty for plotting.")
                        # current_template_for_plot is already an empty array
                    
                    plot_template_and_overlaid_pulses(
                        signal_to_plot=signal_for_plot,
                        times_array=raw.times,
                        pulse_starts_samples=pulse_starts_abs.astype(int) if pulse_starts_abs is not None else np.array([]),
                        mean_signal_overall=avg_signal_full, # Pass the overall mean signal
                        pulse_ends_samples=pulse_ends_abs.astype(int) if pulse_ends_abs is not None else np.array([]),
                        template_waveform=current_template_for_plot,
                        sfreq=sampleRate,
                        channel_name=ch_name_for_plot,
                        stim_freq_hz=final_est_stim_freq,
                        seed_peak_abs_sample=seed_peak_abs_for_plot
                    )

                    # Now plot the signal and its derivatives with the detected pulses
                    if 'avg_signal_full' in locals() and 'raw' in locals() and hasattr(raw, 'times'):
                        plot_signal_and_derivatives(
                            signal_data=avg_signal_full, # Or signal_for_plot if you prefer the channel data
                            times_array=raw.times,
                            signal_name="Average Signal", # Or ch_name_for_plot
                            stim_freq_hz=final_est_stim_freq,
                            pulse_starts_samples=pulse_starts_abs.astype(int) if pulse_starts_abs is not None else np.array([]),
                            pulse_ends_samples=pulse_ends_abs.astype(int) if pulse_ends_abs is not None else np.array([]),
                            sfreq=sampleRate
                        )
                    else:
                        print("Could not plot signal and derivatives due to missing data (avg_signal_full or raw.times).")
                else: # No pulses identified (pulse_starts_rel is None or empty)
                    print("No pulses identified by automatic function, or template generation failed. Skipping full results plot.")
                    # You could optionally plot just the raw signal here if desired, e.g.:
                    # plt.figure(figsize=(15,5))
                    # plt.plot(raw.times, avg_signal_full)
                    # plt.title("Average Signal (No Pulses Detected for Plotting)")
                    # plt.show()
            else: # if len(signal_for_template) == 0
                print("Signal for template matching is empty, skipping pulse identification and plotting.")
        else: # if not final_est_stim_freq
            print("No stimulation frequency estimated, skipping pulse identification and plotting.")

        # Show all generated Matplotlib figures at the end
        plt.show()

    except SystemExit:
        print("Program execution cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")