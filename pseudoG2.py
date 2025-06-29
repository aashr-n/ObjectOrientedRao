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
    title="Select an EEG .fif file",
    filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
    )
    
if not file_path:
    print("No file was selected. Exiting the program.")
    raise SystemExit        
print(f"Loading selected file: {file_path}")

# Load the data into memory for processing
raw = mne.io.read_raw_fif(file_path, preload=True)

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


# --- 6. Template Matching for Stim Pulse Identification (inspired by old.py) ---
def identify_stim_pulses_template_matching(
    avg_signal,
    sfreq,
    stim_freq_hz,
    template_half_width_as_factor_of_period, # New: e.g., 0.25 for a quarter period on each side
    template_center_offset_as_factor_of_period, # New: e.g., 0.1 to shift window 10% of period earlier
    num_pulses_for_template=15, # Increased default slightly
    prioritize_sharper_spikes=True, # New: Flag to enable sharpness prioritization
    scan_initial_window_for_sharpest_peak=True, # New: Scan initial part of signal for a seed peak
    initial_scan_duration_factor=2.0, # New: e.g., 2.0 for 2/stim_freq_hz seconds
    sharpness_window_samples=3, # New: Samples on each side of peak to calculate sharpness
    initial_peak_prominence_factor=3.0, # Multiplier for std dev for initial threshold
    initial_peak_dist_factor=0.8, # Factor of stim period for min_dist
    mf_peak_percentile=80,
    mf_peak_dist_factor=0.5 # Factor of stim period for distance
):
    """
    Identifies stimulation pulses using template matching.

    Args:
        avg_signal (np.ndarray): Average signal across channels (1D).
        sfreq (float): Sampling frequency in Hz.
        stim_freq_hz (float): Estimated stimulation frequency in Hz.
        
        template_half_width_as_factor_of_period (float): Factor of the stimulation period
            to determine the half-width of the template window. E.g., 0.5 means
            the template half-width is 0.5 * stim_period.
        template_center_offset_as_factor_of_period (float): Factor of the stimulation
            period to offset the center of the template extraction window relative
            to the detected peak. Positive values shift the window earlier.
        num_pulses_for_template (int): Number of most prominent initial pulses to
            average for creating the template.
        prioritize_sharper_spikes (bool): If True, prioritizes sharper spikes (based on local
            derivative) in addition to amplitude for template selection.
        scan_initial_window_for_sharpest_peak (bool): If True, the function will scan an initial
            window of the provided `avg_signal` (duration based on `initial_scan_duration_factor`)
            to find the sharpest peak, which will be prioritized for template inclusion.
        initial_scan_duration_factor (float): Multiplier for (1/stim_freq_hz) to define the duration
            of the initial scan window. Only used if `scan_initial_window_for_sharpest_peak` is True.
        sharpness_window_samples (int): Number of samples to look on each side of a peak
            to calculate its sharpness. Only used if prioritize_sharper_spikes is True.
        initial_peak_prominence_factor (float): Factor to multiply std dev by for
            initial peak thresholding.
        initial_peak_dist_factor (float): Factor of stimulation period for minimum
            distance between initial peaks.
        mf_peak_percentile (int): Percentile for matched-filter output peak
            detection threshold.
        mf_peak_dist_factor (float): Factor of stimulation period for minimum
            distance between matched-filter peaks.

    Returns:
        tuple: (pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output)
               Returns (None, None, None, None) if steps fail.
    """
    print(f"\n--- Identifying stimulation pulses using template matching (stim_freq: {stim_freq_hz:.2f} Hz) ---")

    if stim_freq_hz <= 0:
        print("Error: stim_freq_hz must be positive.")
        return None, None, None, None
    stim_period_s = 1.0 / stim_freq_hz

    # 1. Template Window Sizing (New logic)
    template_window_samples = int(template_half_width_as_factor_of_period * stim_period_s * sfreq)
    if template_window_samples <= 0:
        print(f"Warning: Calculated template_window_samples is {template_window_samples}. "
              f"This might be due to small template_half_width_as_factor_of_period ({template_half_width_as_factor_of_period}) "
              f"or low stim_freq_hz/sfreq. Setting to 1 sample.")
        template_window_samples = 1 # Ensure a minimal valid window
    template_total_len = 2 * template_window_samples
    print(f"Template params: half-width={template_window_samples} samples, total_len={template_total_len} samples.")

    # 2. Template Centering Offset (New logic)
    template_center_offset_samples = int(template_center_offset_as_factor_of_period * stim_period_s * sfreq)
    print(f"Template center offset: {template_center_offset_samples} samples (earlier).")

    the_sharpest_seed_peak_idx = None
    # --- Step 0 (New): Scan Initial Window for the Sharpest Peak (if enabled) ---
    if scan_initial_window_for_sharpest_peak and stim_freq_hz > 0:
        scan_duration_s = initial_scan_duration_factor / stim_freq_hz
        scan_end_sample = min(int(scan_duration_s * sfreq), len(avg_signal))
        print(f"Step 0: Scanning initial {scan_duration_s:.3f}s (up to sample {scan_end_sample}) of the provided signal for a sharp seed peak.")

        if scan_end_sample > sharpness_window_samples * 2: # Ensure window is large enough for peak finding and sharpness
            initial_signal_segment = avg_signal[:scan_end_sample]
            # Looser threshold for initial scan to find candidates
            scan_thresh = np.mean(np.abs(initial_signal_segment)) + 1.5 * np.std(np.abs(initial_signal_segment))
            peaks_in_scan, _ = find_peaks(np.abs(initial_signal_segment), height=scan_thresh)

            if len(peaks_in_scan) > 0:
                scan_sharpness_scores = []
                valid_scan_peaks = []
                for p_idx_relative in peaks_in_scan:
                    if p_idx_relative >= sharpness_window_samples and \
                       p_idx_relative < len(initial_signal_segment) - sharpness_window_samples:
                        sharpness = np.abs(initial_signal_segment[p_idx_relative] - initial_signal_segment[p_idx_relative - 1]) + \
                                    np.abs(initial_signal_segment[p_idx_relative] - initial_signal_segment[p_idx_relative + 1])
                        scan_sharpness_scores.append(sharpness)
                        valid_scan_peaks.append(p_idx_relative)
                
                if scan_sharpness_scores:
                    the_sharpest_seed_peak_idx = valid_scan_peaks[np.argmax(scan_sharpness_scores)]
                    print(f"Sharpest seed peak in initial scan found at relative sample: {the_sharpest_seed_peak_idx} within the scanned segment.")
                else:
                    print("No valid peaks for sharpness calculation in the initial scan window of the signal.")
            else:
                print("No peaks found in the initial scan window of the signal.")
        else:
            print("Initial scan window too short based on signal length or sharpness_window_samples. Skipping initial scan.")


    # --- Step 1: Initial rough detection of artifact starts (Original logic) ---

    print("Step 1: Initial rough pulse detection...")
    thresh_init = np.mean(np.abs(avg_signal)) + initial_peak_prominence_factor * np.std(np.abs(avg_signal))
    min_dist_samples = int(initial_peak_dist_factor * (sfreq / stim_freq_hz))
    
    initial_peaks_indices, _ = find_peaks(np.abs(avg_signal), height=thresh_init, distance=min_dist_samples)
    
    if len(initial_peaks_indices) == 0:
        print("No initial pulses found. Cannot proceed with template matching.")
        return None, None, None, None
    print(f"Found {len(initial_peaks_indices)} initial candidate pulses.")


    # --- Step 2a: Selection of Prominent Pulses for Template (New logic) ---
    print(f"Step 2a: Selecting up to {num_pulses_for_template} most prominent initial pulses for template...")
    
    selected_peak_indices_for_template_list = []

    if len(initial_peaks_indices) > 0:
        if prioritize_sharper_spikes or the_sharpest_seed_peak_idx is not None:
            # Calculate scores for all initial_peaks_indices
            sharpness_scores = []
            peak_amplitudes_for_scoring = []
            valid_initial_peaks = []

            for p_idx in initial_peaks_indices:
                # Ensure we are within bounds for sharpness calculation
                if p_idx >= sharpness_window_samples and p_idx < len(avg_signal) - sharpness_window_samples:
                    # Or, sum of absolute differences to immediate neighbors:
                    sharpness = np.abs(avg_signal[p_idx] - avg_signal[p_idx - 1]) + \
                                np.abs(avg_signal[p_idx] - avg_signal[p_idx + 1])
                    sharpness_scores.append(sharpness)
                    peak_amplitudes_for_scoring.append(np.abs(avg_signal[p_idx]))
                    valid_initial_peaks.append(p_idx)
            
            if not valid_initial_peaks: # Fallback if no peaks allow sharpness calculation
                print("Warning: Could not calculate sharpness for any initial peaks for general selection. Falling back to amplitude only for general selection.")
                peak_amplitudes = np.abs(avg_signal[initial_peaks_indices])
                combined_scores = peak_amplitudes
                peaks_to_score = initial_peaks_indices
            else:
                if prioritize_sharper_spikes:
                    print("Prioritizing sharper spikes for general template selection.")
                    combined_scores = np.array(peak_amplitudes_for_scoring) * np.array(sharpness_scores)
                else: # Only amplitude if prioritize_sharper_spikes is false but seed peak was found
                    combined_scores = np.array(peak_amplitudes_for_scoring)
                peaks_to_score = np.array(valid_initial_peaks)
        else: # Original logic if not prioritizing sharpness and no seed peak
            combined_scores = np.abs(avg_signal[initial_peaks_indices])
            peaks_to_score = initial_peaks_indices

        sorted_score_indices = np.argsort(combined_scores)[::-1]

        # Prioritize the sharpest seed peak if found
        if the_sharpest_seed_peak_idx is not None:
            if the_sharpest_seed_peak_idx in peaks_to_score: # Check if it's among the scorable peaks
                selected_peak_indices_for_template_list.append(the_sharpest_seed_peak_idx)
                print(f"Added sharpest seed peak (sample {the_sharpest_seed_peak_idx}) to template list.")
            else:
                # This might happen if the seed peak was filtered out by initial_peak_dist_factor
                # or other criteria of the main find_peaks. Consider if this is acceptable.
                print(f"Warning: Sharpest seed peak (sample {the_sharpest_seed_peak_idx}) was not in the list of scorable initial peaks. Not adding it.")

        # Fill remaining slots
        num_needed = num_pulses_for_template - len(selected_peak_indices_for_template_list)
        for scored_idx in sorted_score_indices:
            peak = peaks_to_score[scored_idx]
            if num_needed <= 0: break
            if peak not in selected_peak_indices_for_template_list: # Avoid duplicates
                selected_peak_indices_for_template_list.append(peak)
                num_needed -= 1
        selected_peak_indices_for_template = np.array(selected_peak_indices_for_template_list)
    else:
        # This case should not be reached due to the check above, but for safety:
        selected_peak_indices_for_template = np.array([], dtype=int)

    if len(selected_peak_indices_for_template) == 0:
        print("No initial pulses selected for template creation (e.g., num_pulses_for_template might be 0 or no initial pulses found).")
        return None, None, None, None
    print(f"Selected {len(selected_peak_indices_for_template)} pulses to build template.")

    # --- Step 2b: Template Snippet Extraction and Averaging (New logic) ---
    print(f"Step 2b: Building template from selected pulses...")
    snippets = []
    
    # Define a slightly wider window for finding the true peak within each snippet
    # This helps ensure the template_window_samples captures the main morphology
    # even if the initial peak_idx isn't perfectly at the max of the artifact.
    extraction_half_width_for_true_peak = template_window_samples + sharpness_window_samples # Or some other reasonable extension

    for peak_idx in selected_peak_indices_for_template:
        # Initial center for extracting a wider snippet, adjusted by the general offset
        initial_extraction_center = peak_idx - template_center_offset_samples

        # Extract a wider snippet to find the true local peak
        wide_snippet_start = max(0, initial_extraction_center - extraction_half_width_for_true_peak)
        wide_snippet_end = min(len(avg_signal), initial_extraction_center + extraction_half_width_for_true_peak)
        wide_snippet = avg_signal[wide_snippet_start:wide_snippet_end]

        if len(wide_snippet) == 0:
            continue # Should not happen if peak_idx is valid

        # Find the index of the maximum absolute value within this wide snippet
        true_peak_in_wide_snippet_idx = np.argmax(np.abs(wide_snippet))
        # This is the refined center for the final template snippet
        final_extraction_center = wide_snippet_start + true_peak_in_wide_snippet_idx
        
        # Desired start and end of the *final* template snippet in avg_signal coordinates
        desired_start_in_signal = final_extraction_center - template_window_samples
        desired_end_in_signal = final_extraction_center + template_window_samples # Exclusive end for slicing

        # Create an empty snippet of the required total length, filled with zeros
        current_snippet = np.zeros(template_total_len)

        # Determine the actual part of avg_signal to copy
        src_start_in_avg_signal = max(0, desired_start_in_signal)
        src_end_in_avg_signal = min(len(avg_signal), desired_end_in_signal)
        
        # Segment from avg_signal to be copied
        avg_signal_segment_to_copy = avg_signal[src_start_in_avg_signal:src_end_in_avg_signal]
        len_segment_to_copy = len(avg_signal_segment_to_copy)

        if len_segment_to_copy > 0:
            # Determine where in `current_snippet` this data should be placed
            dest_start_in_snippet = max(0, -desired_start_in_signal)
            dest_end_in_snippet = dest_start_in_snippet + len_segment_to_copy
            
            if dest_end_in_snippet <= template_total_len:
                current_snippet[dest_start_in_snippet:dest_end_in_snippet] = avg_signal_segment_to_copy
            else:
                # This should not happen if logic is correct and template_total_len is sufficient
                # It implies the segment to copy is too large for the destination slot
                print(f"Warning: Snippet for peak_idx {peak_idx} had inconsistent lengths. "
                      f"dest_end_in_snippet ({dest_end_in_snippet}) > template_total_len ({template_total_len}). Skipping this snippet part.")
        
        snippets.append(current_snippet)

    if not snippets:
        print("No valid snippets collected for template creation.") # Should be caught earlier
        return None, None, None, None
        
    template_waveform = np.mean(snippets, axis=0)
    if np.all(template_waveform == 0):
        print("Warning: Template is all zeros. Check initial pulse detection, selection, sharpness metric, or signal quality.")
        # Depending on requirements, you might return here or allow to proceed

    # --- Step 3: Matched filter (Original logic) ---
    print("Step 3: Applying matched filter...")
    if len(template_waveform) == 0: # Should not happen if template_total_len > 0
        print("Error: Template waveform is empty.")
        return None, None, None, None
    mf_kernel = template_waveform[::-1]
    matched_filter_output = np.convolve(avg_signal, mf_kernel, mode='same')

    # --- Step 4: Detect peaks in matched-filter output (Original logic) ---
    print("Step 4: Detecting peaks in matched-filter output...")
    stim_period_samples = sfreq / stim_freq_hz # recalculate for clarity or use stim_period_s * sfreq
    mf_peak_threshold = np.percentile(matched_filter_output, mf_peak_percentile)
    mf_distance_samples = int(mf_peak_dist_factor * stim_period_samples)
    
    # These are the centers of the detected template matches
    detected_pulse_centers, _ = find_peaks(
        matched_filter_output,
        height=mf_peak_threshold,
        distance=mf_distance_samples
    )

    if len(detected_pulse_centers) == 0:
        print("No pulses found after matched filtering.")
        return detected_pulse_centers, None, template_waveform, matched_filter_output

    # Adjust detected pulse centers to define start and end of the artifact window
    # The detected_pulse_centers are where the center of the kernel (reversed template) aligns for a peak.
    # template_window_samples is the half-length of the template.
    # template_total_len is the full length of the template.
    
    pulse_starts_samples = [max(0, center - template_window_samples) for center in detected_pulse_centers]
    pulse_ends_samples = [min(len(avg_signal) - 1, start + template_total_len - 1) for start in pulse_starts_samples]


    print(f"Identified {len(pulse_starts_samples)} stimulation pulses via template matching.")
    return pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output


# --- 6b. Template Matching V2 (for complex, multi-spike artifacts) ---
def identify_stim_pulses_template_matching_v2(
    avg_signal,
    sfreq,
    stim_freq_hz,
    template_start_ms_before_peak=5.0, # NEW: How many ms to look BACK from the detected peak ## how many samples I think actually!!!
    template_end_ms_after_peak=5.0,   # NEW: How many ms to look FORWARD from the detected peak
    num_pulses_for_template=20,
    initial_peak_prominence_factor=2.5, # Adjusted default
    initial_peak_dist_factor=0.8,
    mf_peak_percentile=85, # Adjusted default
    mf_peak_dist_factor=0.5
):
    """
    Identifies stimulation pulses using template matching, specifically designed for
    complex, multi-peak artifacts by using an ASYMMETRIC window.

    Instead of a symmetric half-width, this function defines the template window
    based on milliseconds before and after the detected primary peak.

    Args:
        avg_signal (np.ndarray): Average signal across channels (1D).
        sfreq (float): Sampling frequency in Hz.
        stim_freq_hz (float): Estimated stimulation frequency in Hz.
        template_start_ms_before_peak (float): The duration in milliseconds to include
            in the template BEFORE the detected peak. Crucial for capturing leading spikes.
        template_end_ms_after_peak (float): The duration in milliseconds to include
            in the template AFTER the detected peak.
        num_pulses_for_template (int): Number of most prominent initial pulses to
            average for creating the template.
        initial_peak_prominence_factor (float): Factor to multiply std dev by for
            initial peak thresholding.
        initial_peak_dist_factor (float): Factor of stimulation period for minimum
            distance between initial peaks.
        mf_peak_percentile (int): Percentile for matched-filter output peak
            detection threshold.
        mf_peak_dist_factor (float): Factor of stimulation period for minimum
            distance between matched-filter peaks.

    Returns:
        tuple: (pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output)
    """
    print(f"\n--- Identifying complex stimulation pulses using Template Matching V2 (stim_freq: {stim_freq_hz:.2f} Hz) ---")

    if stim_freq_hz <= 0:
        print("Error: stim_freq_hz must be positive.")
        return None, None, None, None

    # 1. Convert MS-based window to samples
    samples_before_peak = int((template_start_ms_before_peak / 1000.0) * sfreq)
    samples_after_peak = int((template_end_ms_after_peak / 1000.0) * sfreq)
    template_total_len = samples_before_peak + samples_after_peak
    
    if template_total_len <= 0:
        print("Error: Template length is zero. Check template window parameters.")
        return None, None, None, None
    
    print(f"Template Window: {samples_before_peak} samples before peak, {samples_after_peak} samples after.")
    print(f"Total template length: {template_total_len} samples.")


    # 2. Initial rough detection of the LARGEST spike in each artifact complex
    print("Step 1: Initial rough pulse detection (finding primary peak of each complex)...")
    thresh_init = np.mean(np.abs(avg_signal)) + initial_peak_prominence_factor * np.std(np.abs(avg_signal))
    min_dist_samples = int(initial_peak_dist_factor * (sfreq / stim_freq_hz))
    
    # We search on the absolute signal to robustly find the main peak regardless of polarity
    initial_peaks_indices, _ = find_peaks(np.abs(avg_signal), height=thresh_init, distance=min_dist_samples)
    
    if len(initial_peaks_indices) == 0:
        print("No initial pulses found. Cannot proceed. Try lowering 'initial_peak_prominence_factor'.")
        return None, None, None, None
    print(f"Found {len(initial_peaks_indices)} initial candidate pulses.")

    # 3. Select the most prominent pulses to build the template
    print(f"Step 2: Selecting up to {num_pulses_for_template} most prominent pulses for template...")
    if len(initial_peaks_indices) > num_pulses_for_template:
        # Sort by amplitude and take the top N
        peak_amplitudes = np.abs(avg_signal[initial_peaks_indices])
        strongest_peak_indices = np.argsort(peak_amplitudes)[::-1][:num_pulses_for_template]
        selected_peak_indices_for_template = initial_peaks_indices[strongest_peak_indices]
    else:
        selected_peak_indices_for_template = initial_peaks_indices
    
    print(f"Selected {len(selected_peak_indices_for_template)} pulses to build template.")

    # 4. Build the template using the ASYMMETRIC window
    print(f"Step 3: Building template from selected pulses...")
    snippets = []
    for peak_idx in selected_peak_indices_for_template:
        # Define the start and end of the asymmetric window
        window_start = peak_idx - samples_before_peak
        window_end = peak_idx + samples_after_peak

        # Handle edges of the signal gracefully by padding with zeros
        snippet = np.zeros(template_total_len)
        
        # Determine the source segment from the actual signal
        src_start = max(0, window_start)
        src_end = min(len(avg_signal), window_end)
        
        # Determine the destination segment in our snippet array
        dest_start = max(0, -window_start)
        dest_end = dest_start + (src_end - src_start)
        
        if dest_end > template_total_len:
            # This can happen if the last snippet goes past the array boundary
            # and our calculation overflows. We trim it.
            dest_end = template_total_len
            src_end = src_start + (dest_end - dest_start)

        # Copy the data
        if dest_start < dest_end:
             snippet[dest_start:dest_end] = avg_signal[src_start:src_end]

        snippets.append(snippet)

    if not snippets:
        print("No valid snippets collected.")
        return None, None, None, None
        
    template_waveform = np.mean(snippets, axis=0)
    
    # 5. Matched filter
    print("Step 4: Applying matched filter...")
    mf_kernel = template_waveform[::-1]
    matched_filter_output = np.convolve(avg_signal, mf_kernel, mode='same')

    # 6. Detect peaks in matched-filter output
    print("Step 5: Detecting final peaks in matched-filter output...")
    stim_period_samples = sfreq / stim_freq_hz
    mf_peak_threshold = np.percentile(matched_filter_output, mf_peak_percentile)
    mf_distance_samples = int(mf_peak_dist_factor * stim_period_samples)
    
    detected_pulse_centers, _ = find_peaks(
        matched_filter_output,
        height=mf_peak_threshold,
        distance=mf_distance_samples
    )

    if len(detected_pulse_centers) == 0:
        print("No pulses found after matched filtering. Try lowering 'mf_peak_percentile'.")
        return None, None, template_waveform, matched_filter_output

    # The 'start' of the pulse corresponds to the beginning of our template window.
    # The detected center corresponds to the location of the primary peak used for anchoring.
    pulse_starts_samples = detected_pulse_centers - samples_before_peak
    pulse_ends_samples = detected_pulse_centers + samples_after_peak

    print(f"Identified {len(pulse_starts_samples)} stimulation pulses via template matching.")
    return pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output


# --- 6c. Template Matching V3 (Spikiest Peak at Epoch Onset) ---
def identify_stim_pulses_template_matching_v3(
    avg_signal_epoch,
    sfreq,
    stim_freq_hz,
    template_start_ms_before_peak=7.0,
    template_end_ms_after_peak=7.0,
    seed_search_duration_factor=2.0,
    sharpness_search_prominence=1.5
):
    """
    Identifies stimulation pulses by creating a template from a SINGLE, "spikiest"
    peak found at the very beginning of the provided signal (epoch).

    Args:
        avg_signal_epoch (np.ndarray): The signal segment corresponding to the
            stimulation epoch. The search for the spiky peak will start from index 0.
        sfreq (float): Sampling frequency in Hz.
        stim_freq_hz (float): Estimated stimulation frequency in Hz.
        template_start_ms_before_peak (float): Duration in ms to include in the
            template BEFORE the spikiest peak.
        template_end_ms_after_peak (float): Duration in ms to include in the
            template AFTER the spikiest peak.
        seed_search_duration_factor (float): Multiplier for the stimulation period
            (1/stim_freq_hz) to define the search window duration.
            Default of 2.0 means it searches in the first 2/stim_freq_hz seconds.
        sharpness_search_prominence (float): The prominence factor (multiplier of std dev)
            to find candidate peaks for the sharpness test.

    Returns:
        tuple: (pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output)
    """
    print(f"\n--- Template Matching V3: Finding spikiest peak at epoch onset (stim_freq: {stim_freq_hz:.2f} Hz) ---")

    # --- Step 1: Find the "spikiest" peak in the initial part of the epoch ---
    search_duration_s = seed_search_duration_factor / stim_freq_hz
    search_end_sample = min(int(search_duration_s * sfreq), len(avg_signal_epoch))
    
    print(f"Searching for spikiest peak in the first {search_duration_s:.3f}s (up to sample {search_end_sample}) of the epoch.")
    
    search_segment = avg_signal_epoch[:search_end_sample]

    if len(search_segment) < 3: # Need at least 3 samples to find a peak
        print("Epoch segment too short to find a seed peak. Cannot proceed.")
        return None, None, None, None

    # Find all candidate peaks in this initial window
    thresh = np.mean(np.abs(search_segment)) + sharpness_search_prominence * np.std(np.abs(search_segment))
    candidate_peaks, _ = find_peaks(np.abs(search_segment), height=thresh)

    if len(candidate_peaks) == 0:
        print(f"No candidate peaks found in the initial search window. Try lowering 'sharpness_search_prominence'.")
        return None, None, None, None

    # Calculate sharpness for each candidate and find the spikiest
    max_sharpness = -1
    spikiest_peak_idx = -1

    for p_idx in candidate_peaks:
        # Ensure the peak is not at the very edge to allow sharpness calculation
        if 1 <= p_idx < len(search_segment) - 1:
            # Sharpness = sum of absolute slopes on either side of the peak
            sharpness = np.abs(search_segment[p_idx] - search_segment[p_idx - 1]) + \
                        np.abs(search_segment[p_idx] - search_segment[p_idx + 1])
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                spikiest_peak_idx = p_idx

    if spikiest_peak_idx == -1:
        print("Could not determine a spikiest peak (e.g., all candidates were at the signal edge).")
        return None, None, None, None

    print(f"Found spikiest seed peak at sample {spikiest_peak_idx} within the epoch.")

    # --- Step 2: Build the template from this single peak ---
    print("Step 2: Building template from the single spikiest peak...")
    samples_before_peak = int((template_start_ms_before_peak / 1000.0) * sfreq)
    samples_after_peak = int((template_end_ms_after_peak / 1000.0) * sfreq)
    template_total_len = samples_before_peak + samples_after_peak

    window_start = spikiest_peak_idx - samples_before_peak
    window_end = spikiest_peak_idx + samples_after_peak
    
    # Extract the snippet, padding with zeros at the edges
    template_waveform = np.zeros(template_total_len)
    src_start = max(0, window_start)
    src_end = min(len(avg_signal_epoch), window_end)
    dest_start = max(0, -window_start)
    dest_end = dest_start + (src_end - src_start)
    
    if dest_start < dest_end:
        template_waveform[dest_start:dest_end] = avg_signal_epoch[src_start:src_end]

    # --- Step 3: Apply Matched Filter and Detect All Pulses ---
    print("Step 3: Applying matched filter across the entire epoch...")
    mf_kernel = template_waveform[::-1]
    matched_filter_output = np.convolve(avg_signal_epoch, mf_kernel, mode='same')

    print("Step 4: Detecting final peaks...")
    stim_period_samples = sfreq / stim_freq_hz
    mf_peak_percentile = 85 # A good starting point
    mf_distance_samples = int(0.5 * stim_period_samples) # 50% of period to avoid double detection
    mf_peak_threshold = np.percentile(matched_filter_output, mf_peak_percentile)
    
    detected_pulse_centers, _ = find_peaks(
        matched_filter_output,
        height=mf_peak_threshold,
        distance=mf_distance_samples
    )

    if len(detected_pulse_centers) == 0:
        print("No pulses found after matched filtering.")
        return None, None, template_waveform, matched_filter_output

    pulse_starts_samples = detected_pulse_centers - samples_before_peak
    pulse_ends_samples = detected_pulse_centers + samples_after_peak

    print(f"Identified {len(pulse_starts_samples)} stimulation pulses.")
    return pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output

# --- 6. Fully Automatic Pulse Identification Function ---
def find_stimulation_pulses_auto(
    avg_signal_epoch,
    sfreq,
    stim_freq_hz
):
    """
    Automatically detects stimulation pulses with varying numbers of components
    without requiring manual parameter tuning for the template shape.

    It works by:
    1. Finding a single "spiky" seed peak at the epoch onset.
    2. Analyzing the signal around this peak to dynamically find the start and
       end boundaries of the entire multi-component artifact.
    3. Creating a template from these dynamic boundaries.
    4. Using this template in a matched filter to find all other pulses.

    Args:
        avg_signal_epoch (np.ndarray): The signal segment of the stimulation epoch.
        sfreq (float): Sampling frequency.
        stim_freq_hz (float): Estimated stimulation frequency.

    Returns:
        tuple: (pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output)
    """
    print("\n--- Running Fully Automatic Pulse Detection ---")

    # --- Step 1: Find a single "spiky" seed peak to anchor our analysis ---
    search_duration_s = 2.0 / stim_freq_hz
    search_end_sample = min(int(search_duration_s * sfreq), len(avg_signal_epoch))
    search_segment = avg_signal_epoch[:search_end_sample]

    if len(search_segment) < 3:
        print("Automatic Strategy: Epoch segment too short to find a seed peak.")
        return None, None, None, None

    # Find candidate peaks to test for "spikiness"
    thresh = np.mean(np.abs(search_segment)) + 1.5 * np.std(np.abs(search_segment))
    candidate_peaks, _ = find_peaks(np.abs(search_segment), height=thresh)

    if not candidate_peaks.any():
        print("Automatic Strategy: No candidate seed peaks found.")
        return None, None, None, None

    # Find the spikiest among the candidates
    sharpness_scores, valid_peaks = [], []
    for p_idx in candidate_peaks:
        if 1 <= p_idx < len(search_segment) - 1:
            sharpness = np.abs(search_segment[p_idx] - search_segment[p_idx - 1]) + np.abs(search_segment[p_idx] - search_segment[p_idx + 1])
            sharpness_scores.append(sharpness)
            valid_peaks.append(p_idx)
    
    if not valid_peaks:
        print("Automatic Strategy: No valid peaks for sharpness calculation.")
        return None, None, None, None
        
    seed_peak_location = valid_peaks[np.argmax(sharpness_scores)]
    print(f"Found spiky seed peak at sample {seed_peak_location}.")

    # --- Step 2: Dynamically determine the artifact's boundaries ---
    # Define an analysis window around the seed peak (one full stimulation period)
    period_samples = int(sfreq / stim_freq_hz)
    analysis_win_start = max(0, seed_peak_location - period_samples // 2)
    analysis_win_end = min(len(avg_signal_epoch), seed_peak_location + period_samples // 2)
    analysis_snippet = avg_signal_epoch[analysis_win_start:analysis_win_end]

    # Find ALL bumps and dips within this window to find the true start and end
    feature_thresh = 1.0 * np.std(analysis_snippet) # Lenient threshold
    peaks, _ = find_peaks(analysis_snippet, height=feature_thresh, distance=2)
    dips, _ = find_peaks(-analysis_snippet, height=feature_thresh, distance=2)
    
    all_features = np.concatenate((peaks, dips))
    if not all_features.any():
        print("Automatic Strategy: Could not find features to define artifact boundaries.")
        return None, None, None, None
        
    dynamic_start = np.min(all_features)
    dynamic_end = np.max(all_features)
    
    # --- Step 3: Create the template using these dynamic boundaries ---
    # The template is the signal between the first and last feature
    template_dynamic = analysis_snippet[dynamic_start:dynamic_end + 1]
    
    # For robust matching, we place this dynamic template onto a fixed-size canvas
    template_canvas = np.zeros(period_samples)
    
    # Place it in the center of the canvas
    start_offset_on_canvas = (period_samples - len(template_dynamic)) // 2
    if start_offset_on_canvas < 0: start_offset_on_canvas = 0
        
    end_offset_on_canvas = start_offset_on_canvas + len(template_dynamic)
    if end_offset_on_canvas > period_samples: end_offset_on_canvas = period_samples

    template_canvas[start_offset_on_canvas:end_offset_on_canvas] = template_dynamic[:len(template_canvas[start_offset_on_canvas:end_offset_on_canvas])]

    template_waveform = template_canvas
    print(f"Dynamically created template of length {len(template_dynamic)} samples (on a {period_samples}-sample canvas).")

    # --- Step 4: Matched filter and final detection ---
    mf_kernel = template_waveform[::-1]
    matched_filter_output = np.convolve(avg_signal_epoch, mf_kernel, mode='same')

    mf_peak_threshold = np.percentile(matched_filter_output, 85)
    mf_distance_samples = int(0.6 * period_samples)
    
    detected_pulse_centers, _ = find_peaks(
        matched_filter_output, height=mf_peak_threshold, distance=mf_distance_samples
    )

    if not detected_pulse_centers.any():
        print("Automatic Strategy: No pulses found after matched filtering.")
        return None, None, template_waveform, matched_filter_output

    half_canvas = len(template_waveform) // 2
    pulse_starts_samples = detected_pulse_centers - half_canvas
    pulse_ends_samples = detected_pulse_centers + half_canvas

    print(f"Identified {len(pulse_starts_samples)} stimulation pulses automatically.")
    return pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output


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

        # 2. Refine Frequency Estimate using the Strongest Channel
        final_est_stim_freq = None
        strong_channel_idx_for_viz = None
        if estStimFrequency is not None:
            print(f"\nInitial estimated stimulation frequency (from mean PSD): {estStimFrequency:.2f} Hz")
            channel_relative_prominences = [calculate_single_peak_relative_prominence(psds[i, :], freqs, estStimFrequency, neighborhood_hz_rule=1.0) for i in range(n_channels)]
            if channel_relative_prominences:
                strong_channel_idx = np.argmax(channel_relative_prominences)
                strong_channel_idx_for_viz = strong_channel_idx
                print(f"Channel with strongest relative prominence: {raw.ch_names[strong_channel_idx]}")
                final_est_stim_freq = find_prominent_peaks(psd_values=psds[strong_channel_idx, :], frequencies=freqs, prominence=0.1, distance=20)
                if not final_est_stim_freq:
                    final_est_stim_freq = estStimFrequency # Fallback
            else:
                final_est_stim_freq = estStimFrequency
            print(f"Final estimated stimulation frequency: {final_est_stim_freq:.2f} Hz")
        else:
            print("No initial stimulation frequency found.")

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
            signal_for_template, offset_samples = (avg_signal_full[int(epoch_start_s*sampleRate):int(epoch_end_s*sampleRate)], int(epoch_start_s*sampleRate)) if epoch_start_s is not None else (avg_signal_full, 0)
            
            if len(signal_for_template) > 0:
                # Call the final, automatic function.
                pulse_starts_rel, pulse_ends_rel, template, mf_out = find_stimulation_pulses_auto(
                    avg_signal_epoch=signal_for_template,
                    sfreq=sampleRate,
                    stim_freq_hz=final_est_stim_freq
                )

                # 5. Plot the Final Results
                if pulse_starts_rel is not None:
                    pulse_starts_abs = np.array(pulse_starts_rel) + offset_samples
                    pulse_ends_abs = np.array(pulse_ends_rel) + offset_samples
                    mf_full = np.convolve(avg_signal_full, template[::-1], mode='same')
                    
                    signal_for_plot = data[strong_channel_idx_for_viz] if strong_channel_idx_for_viz is not None else avg_signal_full
                    ch_name_for_plot = raw.ch_names[strong_channel_idx_for_viz] if strong_channel_idx_for_viz is not None else "Average"

                    # Assuming you have a plot_pulse_identification_results function defined elsewhere
                    # If not, you'd need to add its definition back in.
                    # For now, let's assume it exists.
                    # plot_pulse_identification_results(...) # You would call your plotting function here.
                    print("Plotting of results would occur here.")


    except SystemExit:
        print("Program execution cancelled by user.")
    except NameError as e:
        # This will catch if the plotting function is missing, which it is in the provided snippet.
        if 'plot_pulse_identification_results' in str(e):
             print("\nDIAGNOSIS: The function 'plot_pulse_identification_results' is not defined.")
             print("Please ensure the definition for your plotting function is included in the script.")
        else:
             print(f"A NameError occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")