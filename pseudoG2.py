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


# --- 6. Template Matching for Stim Pulse Identification (inspired by old.py) ---
def identify_stim_pulses_template_matching(
    avg_signal,
    sfreq,
    stim_freq_hz,
    template_window_ms=5,
    num_pulses_for_template=10,
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
        template_window_ms (int): Half-width of the window for creating template snippets, in ms.
        num_pulses_for_template (int): Number of initial pulses to average for template.
        initial_peak_prominence_factor (float): Factor to multiply std dev by for initial peak threshold.
        initial_peak_dist_factor (float): Factor of stimulation period for minimum distance between initial peaks.
        mf_peak_percentile (int): Percentile for matched-filter output peak detection threshold.
        mf_peak_dist_factor (float): Factor of stimulation period for minimum distance between matched-filter peaks.

    Returns:
        tuple: (pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output)
               Returns (None, None, None, None) if steps fail.
    """
    print(f"\n--- Identifying stimulation pulses using template matching (stim_freq: {stim_freq_hz:.2f} Hz) ---")

    # 1. Initial rough detection of artifact starts
    print("Step 1: Initial rough pulse detection...")
    thresh_init = np.mean(np.abs(avg_signal)) + initial_peak_prominence_factor * np.std(np.abs(avg_signal))
    min_dist_samples = int(initial_peak_dist_factor * (sfreq / stim_freq_hz))
    
    initial_peaks_indices, _ = find_peaks(np.abs(avg_signal), height=thresh_init, distance=min_dist_samples)
    
    if len(initial_peaks_indices) == 0:
        print("No initial pulses found. Cannot proceed with template matching.")
        return None, None, None, None
    print(f"Found {len(initial_peaks_indices)} initial candidate pulses.")

    # 2. Build the template
    print(f"Step 2: Building template from first {min(num_pulses_for_template, len(initial_peaks_indices))} pulses...")
    template_window_samples = int(template_window_ms * sfreq / 1000)
    snippets = []
    for idx in initial_peaks_indices[:num_pulses_for_template]:
        start = max(0, idx - template_window_samples)
        end = min(len(avg_signal), idx + template_window_samples)
        snippet = avg_signal[start:end]
        # Pad/truncate to uniform length (2*template_window_samples)
        required_len = 2 * template_window_samples
        if len(snippet) < required_len:
            padding = required_len - len(snippet)
            # Smart padding: if snippet is from beginning, pad at end; if from end, pad at start
            if start == 0: # Snippet was truncated at the beginning
                 snippet = np.pad(snippet, (0, padding), mode='constant', constant_values=0)
            else: # Snippet was truncated at the end or is shorter for other reasons
                 snippet = np.pad(snippet, (0, padding), mode='constant', constant_values=0)
        elif len(snippet) > required_len:
            snippet = snippet[:required_len] # Should not happen if idx is centered
        
        if len(snippet) == required_len: # Ensure snippet is correct length before appending
            snippets.append(snippet)
        else:
            print(f"Warning: Snippet for index {idx} has unexpected length {len(snippet)}, expected {required_len}. Skipping.")


    if not snippets:
        print("No valid snippets collected for template creation.")
        return None, None, None, None
        
    template_waveform = np.mean(snippets, axis=0)
    if np.all(template_waveform == 0):
        print("Warning: Template is all zeros. Check initial pulse detection or signal quality.")
        # return None, None, None, None # Or allow to proceed if this is expected in some cases

    # 3. Matched filter: convolve with time-reversed template
    print("Step 3: Applying matched filter...")
    mf_kernel = template_waveform[::-1]
    matched_filter_output = np.convolve(avg_signal, mf_kernel, mode='same')

    # 4. Detect peaks in matched-filter output
    print("Step 4: Detecting peaks in matched-filter output...")
    stim_period_samples = sfreq / stim_freq_hz
    mf_peak_threshold = np.percentile(matched_filter_output, mf_peak_percentile)
    mf_distance_samples = int(mf_peak_dist_factor * stim_period_samples)
    
    pulse_starts_samples, _ = find_peaks(
        matched_filter_output,
        height=mf_peak_threshold,
        distance=mf_distance_samples
    )

    if len(pulse_starts_samples) == 0:
        print("No pulses found after matched filtering.")
        return pulse_starts_samples, None, template_waveform, matched_filter_output # Return empty starts and template

    # Define pulse ends based on template length
    template_half_len = len(template_waveform) // 2
    pulse_ends_samples = [min(len(avg_signal) - 1, start + template_half_len) for start in pulse_starts_samples]
    # Adjust pulse_starts based on template alignment (assuming peak of template is center)
    # The convolve 'same' output aligns center of kernel with point.
    # find_peaks gives the peak of the mf_output. If template is symmetric and centered, this is fine.
    # For now, let's assume pulse_starts_samples are the "true" starts relative to template shape.
    # Or, adjust starts to be beginning of template window:
    pulse_starts_samples = [max(0, s - template_half_len) for s in pulse_starts_samples]
    # Recalculate ends to maintain template duration
    pulse_ends_samples = [min(len(avg_signal) - 1, s + len(template_waveform) -1) for s in pulse_starts_samples]


    print(f"Identified {len(pulse_starts_samples)} stimulation pulses via template matching.")
    return pulse_starts_samples, pulse_ends_samples, template_waveform, matched_filter_output




# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        
        #all the  psds are already calculated 
        
        mean_psd = psds.mean(axis=0)
        avg_signal_full = np.mean(data, axis=0) # Calculate average signal for template matching
        
        #this is meant for a 1d array!!!
        estStimFrequency = find_prominent_peaks(
            psd_values=mean_psd,
            frequencies=freqs,
            prominence=0.2,
            distance=50
        )


        # Plot the mean PSD
        fig_psd, ax_psd = plt.subplots(figsize=(10, 6))
        ax_psd.plot(freqs, mean_psd, label='Mean PSD')
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power Spectral Density (V^2/Hz)')
        ax_psd.set_title('Mean PSD Across All Channels')
        if estStimFrequency is not None:
            ax_psd.axvline(estStimFrequency, color='r', linestyle='--', label=f'Est. Stim Freq: {estStimFrequency:.2f} Hz')
        ax_psd.legend()
        ax_psd.grid(True, alpha=0.5)
        plt.show()


        def plot_pulse_identification_results(times_array, signal_to_plot, pulse_starts, pulse_ends,
                                            template, sfreq, mf_output, ch_name=None, stim_freq_hz=None):
            fig_template, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=False)
            fig_template.suptitle(f"Stimulation Pulse Identification Details (Est. Freq: {stim_freq_hz:.2f} Hz)" if stim_freq_hz else "Stimulation Pulse Identification Details", fontsize=16)

            # Plot 1: Template
            template_time = (np.arange(len(template)) - len(template)//2) / sfreq * 1000 # in ms
            axs[0].plot(template_time, template, color='darkorange')
            axs[0].set_title("Derived Artifact Template")
            axs[0].set_xlabel("Time (ms)")
            axs[0].set_ylabel("Amplitude")
            axs[0].grid(True, alpha=0.5)

            # Plot 2: Matched Filter Output (zoomed if many pulses)
            axs[1].plot(times_array, mf_output, color='teal', label='Matched Filter Output')
            if pulse_starts and len(pulse_starts) > 0:
                axs[1].scatter(times_array[pulse_starts], mf_output[pulse_starts], color='red', marker='x', label='Detected Pulses')
                # Zoom around first few pulses if too many
                if len(pulse_starts) > 50: # Arbitrary limit for zooming
                    zoom_end_time = times_array[pulse_starts[min(len(pulse_starts)-1, 10)]] + (10/stim_freq_hz if stim_freq_hz else 0.1)
                    axs[1].set_xlim(times_array[pulse_starts[0]] - (1/stim_freq_hz if stim_freq_hz else 0.01), zoom_end_time)
            axs[1].set_title("Matched Filter Output & Detected Peaks")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Convolution Output")
            axs[1].legend()
            axs[1].grid(True, alpha=0.5)

            # Plot 3: Signal with detected pulses
            axs[2].plot(times_array, signal_to_plot, color='cornflowerblue', label=f'Signal ({ch_name})' if ch_name else 'Signal (Avg)')
            if pulse_starts and pulse_ends:
                for s_idx, e_idx in zip(pulse_starts, pulse_ends):
                    axs[2].axvspan(times_array[s_idx], times_array[e_idx], color='salmon', alpha=0.4, lw=0)
            if pulse_starts: # Add markers for starts for clarity if few pulses
                 axs[2].scatter(times_array[pulse_starts], signal_to_plot[pulse_starts], color='red', marker='|', s=50, label='Pulse Start/End')

            axs[2].set_title(f"Signal with Detected Pulses ({ch_name if ch_name else 'Average'})")
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Amplitude")
            axs[2].legend(loc='upper right')
            axs[2].grid(True, alpha=0.5)
            if pulse_starts and len(pulse_starts) > 0: # Zoom to first epoch of pulses
                start_display_time = times_array[pulse_starts[0]] - (2/stim_freq_hz if stim_freq_hz else 0.02)
                end_display_time = times_array[min(pulse_starts[-1], pulse_starts[0] + int(10*sfreq/stim_freq_hz if stim_freq_hz else sfreq*0.2))] + (2/stim_freq_hz if stim_freq_hz else 0.02)
                axs[2].set_xlim(max(0, start_display_time), min(times_array[-1],end_display_time))


            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()



        #################### THIS PART FINDS WHICH CHANNEL HAS MOST PROMINENT ARTIFACTS #################

        final_est_stim_freq = None # Initialize
        strong_channel_idx_for_viz = None # Initialize for visualization
        if estStimFrequency is not None:
            print(f"\nInitial estimated stimulation frequency (from mean PSD): {estStimFrequency:.2f} Hz")
            
            channel_relative_prominences = []
            search_half_width_hz = 0.5

            print(f"Searching for strongest channel within +/- {search_half_width_hz} Hz of {estStimFrequency:.2f} Hz...")

            for i in range(n_channels):
                channel_psd = psds[i, :]
                
                # Find the actual peak frequency in this channel's PSD within the window
                window_mask = (freqs >= estStimFrequency - search_half_width_hz) & \
                              (freqs <= estStimFrequency + search_half_width_hz)
                
                if not np.any(window_mask):
                    channel_relative_prominences.append(0) # No frequencies in window
                    continue

                local_psd_segment = channel_psd[window_mask]
                local_freqs_segment = freqs[window_mask]
                
                if len(local_psd_segment) == 0:
                    channel_relative_prominences.append(0) # No power values in window
                    continue

                # Find the frequency of max power in this local segment
                actual_peak_freq_in_window = local_freqs_segment[np.argmax(local_psd_segment)]
                
                # Calculate relative prominence for this actual peak in this channel
                # Using a fixed neighborhood for this stage for consistency across channels
                rel_prom = calculate_single_peak_relative_prominence(
                    channel_psd, freqs, actual_peak_freq_in_window, neighborhood_hz_rule=1.0
                )
                channel_relative_prominences.append(rel_prom)

            if channel_relative_prominences: # Check if list is not empty
                strong_channel_idx = np.argmax(channel_relative_prominences)
                max_rel_prom_strong_ch = channel_relative_prominences[strong_channel_idx]
                strong_channel_idx_for_viz = strong_channel_idx # Save for visualization
                print(f"Channel with strongest relative prominence for the peak: {raw.ch_names[strong_channel_idx]} (Index: {strong_channel_idx}, Rel Prom: {max_rel_prom_strong_ch:.2f})")

                # Recalculate stim frequency using only the PSD of the strong channel
                print(f"Recalculating stim frequency using only channel {raw.ch_names[strong_channel_idx]}...")
                final_est_stim_freq = find_prominent_peaks(
                    psd_values=psds[strong_channel_idx, :], # PSD of the strong channel
                    frequencies=freqs,
                    prominence=0.1, # Potentially lower initial prominence for single channel
                    distance=20 # Potentially smaller distance for single channel
                )
                if final_est_stim_freq:
                    print(f"Final estimated stimulation frequency (from strong channel): {final_est_stim_freq:.2f} Hz")
                else:
                    print(f"Could not refine stim frequency from strong channel. Using initial estimate: {estStimFrequency:.2f} Hz")
                    final_est_stim_freq = estStimFrequency # Fallback to initial
            else:
                print("Could not determine strong channel. Using initial estimate.")
                final_est_stim_freq = estStimFrequency # Fallback
        else:
            print("No initial stimulation frequency found. Cannot proceed to find strong channel.")


        # --- Perform Template Matching if a stimulation frequency is found ---
        if final_est_stim_freq is not None:
            pulse_starts, pulse_ends, template, mf_output = identify_stim_pulses_template_matching(
                avg_signal=avg_signal_full, # Use the full average signal
                sfreq=sampleRate,
                stim_freq_hz=final_est_stim_freq
            )

            if pulse_starts is not None and len(pulse_starts) > 0:
                print(f"\nFirst 5 detected pulse start times (s): {np.array(pulse_starts[:5]) / sampleRate}")
                
                signal_for_plot = avg_signal_full
                ch_name_for_plot = "Average Signal"
                if strong_channel_idx_for_viz is not None:
                    signal_for_plot = data[strong_channel_idx_for_viz]
                    ch_name_for_plot = raw.ch_names[strong_channel_idx_for_viz]
                
                plot_pulse_identification_results(
                    raw.times, signal_for_plot, pulse_starts, pulse_ends,
                    template, sampleRate, mf_output, ch_name=ch_name_for_plot, stim_freq_hz=final_est_stim_freq
                )
            else:
                print("Template matching did not yield any pulse detections or failed.")


        ##################### END OF BEST CHANNEL FINDER  ####################


        # Use final_est_stim_freq for visualization if available
        if final_est_stim_freq is not None:
            print(f"\nProceeding with epoch visualization for final estimated frequency: {final_est_stim_freq:.2f} Hz")
            visualize_stim_epoch_with_sliding_window(
                raw, final_est_stim_freq, 
                grace_period_sec=5.0, 
                strong_channel_idx=strong_channel_idx_for_viz
            ) 
        elif estStimFrequency is not None: # Fallback if final_est_stim_freq couldn't be determined but initial was
            print(f"\nProceeding with epoch visualization for initial estimated frequency: {estStimFrequency:.2f} Hz")
            visualize_stim_epoch_with_sliding_window(
                raw, estStimFrequency, 
                grace_period_sec=5.0, 
                strong_channel_idx=strong_channel_idx_for_viz # Still pass if available
            )
        else:
            print("\nNo suitable stimulation frequency found to visualize epoch.")

        



        

    except SystemExit:
        # This catch block prevents a traceback error if the user closes the file dialog.
        print("Program execution cancelled by user.")
