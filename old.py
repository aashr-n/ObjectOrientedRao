#
### Intro
# This is a translation of Kristin Sellers' 2018 artifact rejection protocol
#  [as used in Rao et. al:
#       "Direct Electrical Stimulation of Lateral Orbitofrontal Cortex Acutely
#        Improves Mood in Individuals with Symptoms of Depression.]

# One deviation: this only deals with the "clinical" datatype,
    # though it should not be hard to add functionality

# Another deviation: no manual channel removal here,
    #  that must be done beforehand

# Trying to do this analogous to the original

# ctrl + f "section #" to get to the section beginning

'''table of contents:
    Section 1 : calculating stim rate
    Section 2: Convolution + stim matching
    Section 3: spline removal
    
'''

# To do- analyze section two + three

import mne
import scipy

import os
import numpy as np

from scipy.signal import welch
import matplotlib.pyplot as plt

# make all the gui stuff functions at the end to make it tidy

# Prompt user to select a data file
import tkinter as tk
from tkinter import filedialog
import tkinter.simpledialog as simpledialog


import numpy as np
from scipy.signal import find_peaks

from mne.time_frequency import psd_array_multitaper

#hidden root window for all dialogs
tk_root = tk.Tk()
tk_root.withdraw() 


def getFifFile():
    mne.set_log_level('WARNING')  # only show warnings/errors, not info lines

    file_path = filedialog.askopenfilename(
        title="Select EEG FIF file",
        filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
    )
    if not file_path:
        raise SystemExit("No file selected, exiting.")

    print(f"Selected file: {file_path}")
    print(f"Exists: {os.path.exists(file_path)}")

    #the above lets the user choose a file

    file = mne.io.read_raw_fif(file_path, preload=True)

    return file

raw = getFifFile()

print("File uploaded, fetching stim frequency") #Section 1 starts here




### Parameters first
#  (gonna straight copy from matlab and then move from there)
#   this would be easier to use if you made a GUI for changing these

stimFreq = 100 # in Hz
removeOrReplace = 2 # 1 = replace bad data with zeros; 2 = remove bad data
currentStim = 'OFC_6mA'
# dataType = 1 # 1 = clinical, 2 = TDT, 3 = NeuroOmega . [[THIS PART IS NOT RELEVANT, CLINICAL ONLY]]
rejectStimChans = 1 # 0 = do not reject stim channels; 1 = reject stim channels



sampleRate = raw.info['sfreq']
data = raw.get_data()                # shape (n_chan, n_samples)
n_chan, _ = data.shape

frequencyRes = 0.25 #the divisor defines the
# Increase FFT length for finer frequency resolution (0.01 Hz)
n_fft = int(sampleRate / frequencyRes)


n_channels = len(raw.ch_names)

print(f"The recording has {n_channels} channels.")

# 2) Compute PSD on every channel via multi‐taper
bandwidth = 1  # this is W, T is automatically calculated, L is based off T and W


# --- Compute PSD per channel with progress updates ---
psd_list = []
for idx, ch in enumerate(raw.ch_names):
    print(f"Calculating PSD for channel {idx+1} ({ch})...")
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
print("Multi-taper PSD computation complete for all channels.")

'''
# 2) Compute PSD on every channel via Welch
psd = raw.compute_psd(
    method='welch',
    n_fft=n_fft,
    n_overlap=n_fft // 2,
    n_per_seg=n_fft,
    fmin= frequencyRes,
    fmax= sampleRate / 2 #### lowest 
)
# psd is an instance of PSDArray; extract arrays:
freqs = psd.freqs            # shape (n_freqs,)
psds = psd.get_data()        # shape (n_channels, n_freqs)
'''


# 3) Average across channels
mean_psd = psds.mean(axis=0)



###There was a graph here of the averaged psd, its at the bottom and commented out now

#### Find PEAK

# 1) find peaks and prominences of spectra
peaks, props = find_peaks(mean_psd, prominence=10)# check this, play artound with the prominence
pfreqs = freqs[peaks]
proms  = props['prominences']

#plt.plot(freqs, mean_psd, label = 'mean psd')
#plt.show()

# Print out each peak frequency and its prominence
print("Detected PSD peaks and their prominences:")
for freq_val, prom_val in zip(pfreqs, proms):
    print(f"  {freq_val:.2f} Hz → prominence {prom_val:.4f}")


# Only use the top 20 most prominent peaks
N_display = min(20, len(proms))
top_idx = np.argsort(proms)[::-1][:N_display]
top_freqs_plot = pfreqs[top_idx]
top_proms_plot = proms[top_idx]
top_peaks_plot = peaks[top_idx]

# After you compute the PSD and get freqs:
print("First few frequency bins:", freqs[:5])
res = freqs[1] - freqs[0]
print(f"Frequency resolution = {res:.4f} Hz (constant spacing)")
# Smaller marker size
sizes = top_proms_plot * 100

plt.figure(figsize=(8, 4))
plt.plot(freqs, mean_psd, label='Mean PSD')
plt.scatter(top_freqs_plot, mean_psd[top_peaks_plot], s=sizes, c='red', alpha=0.7)
for f, p in zip(top_freqs_plot, top_proms_plot):
    psd_val = mean_psd[freqs == f][0]
    plt.text(f, psd_val, f"{p:.2f}", ha='center', va='bottom', fontsize=8)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Mean PSD with Peak Prominences')
# Set axis limits to focus on top 10 peaks with margins
x_min = 0
x_max = top_freqs_plot[:10].max() + 20
y_min = 0
y_max = mean_psd[top_peaks_plot[:10]].max() * 1.2
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

from matplotlib.lines import Line2D
# Highlight the stim frequency (lowest freq with prominence > 10)
stim_mask = top_proms_plot > 10
if np.any(stim_mask):
    stim_idx = np.where(stim_mask)[0][np.argmin(top_freqs_plot[stim_mask])]
    stim_freq = top_freqs_plot[stim_idx]
    stim_power = mean_psd[freqs == stim_freq][0]
    plt.scatter([stim_freq], [stim_power], s=top_proms_plot[stim_idx] * 100, c='lime', alpha=0.7)
    stim_legend = Line2D([0], [0], marker='o', color='w', label=f'Stim Frequency ({stim_freq:.1f} Hz)',
                         markerfacecolor='lime', markersize=6, alpha=0.7)
    text_legend = Line2D([], [], color='none', marker='', label='Black numbers = prominence of peak')
    # Update legend handles
    plt.legend(handles=[
        Line2D([], [], color='C0', label='Mean PSD'),
        Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=6, alpha=0.7, label='Peaks'),
        stim_legend,
        text_legend
    ])
else:
    text_legend = Line2D([], [], color='none', marker='', label='Black numbers = prominence of peak')
    plt.legend(handles=[
        Line2D([], [], color='C0', label='Mean PSD'),
        Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=6, alpha=0.7, label='Peaks'),
        text_legend
    ])

plt.tight_layout()
plt.show()

# --- Find peak power near the calculated stim frequency ---
# Define search window ±0.5 Hz around stim_freq
# 1) Build a mask for the ±0.5 Hz window
lower, upper = stim_freq - 0.5, stim_freq + 0.5
mask = (freqs >= lower) & (freqs <= upper)

# 2) Slice out the PSD values in that range
local_psd = mean_psd[mask]
local_freqs = freqs[mask]

# 3) Find the index of the maximum power in that slice
idx = np.argmax(local_psd)

# 4) Extract the corresponding frequency and power
best_freq  = local_freqs[idx]
best_power = local_psd[idx]

print(f"Highest power within ±0.5 Hz of {stim_freq:.2f} Hz:")
print(f"  {best_freq:.2f} Hz with power {best_power:.4e}")



# 2) pick top N by prominence
N = min(10, len(pfreqs)) # chooses smaller between 10 and length of pfreqs array
order = np.argsort(proms)[::-1]
top_freqs = pfreqs[order[:N]]
top_proms = proms[order[:N]]

most_prom_value = top_proms[0]

# 3) compute all adjacent absolute differences and weights
# replace diffs/weights loop with:
diffs = []
weights = []
for i in range(N-1):## this focuses on high prominence frequencies that are next to each other, instead of differences between onces that are further apart
    d = abs(top_freqs[i+1] - top_freqs[i])
    w = top_proms[i] + top_proms[i+1]
    diffs.append(d)
    weights.append(w)
diffs  = np.array(diffs)
weights = np.array(weights)

# diffs: array of all adjacent frequency differences
# weights: corresponding array of weights

# All-pairwise differences & weights
diffs_pairwise = []
weights_pairwise = []
for i in range(N):
    for j in range(i+1, N):
        #k = j - i #index diff between the compared 
        d = abs(top_freqs[j] - top_freqs[i])
        w = (top_proms[i] + top_proms[j]) # k # this makes wider gaps lighter (antiharmonic
        diffs_pairwise.append(d)
        weights_pairwise.append(w)
diffs_pairwise  = np.array(diffs_pairwise)
weights_pairwise = np.array(weights_pairwise)



# Prompt user for desired bin width (Hz) via simple dialog
userStim = simpledialog.askfloat(
    "Stimrate input (X out if unknown, program will estimate)",
    "What's your expected stimulation rate [dont do 95% of expected]",
    initialvalue= 10,
    minvalue=0.0001,
    maxvalue=10000.0,
    parent=tk_root
)

# Prompt user for desired bin width (Hz) via simple dialog
if userStim is None:
    print("No expected stimulation entered, estimating based on average psd")
    bin_width = simpledialog.askfloat(
    "Stim Frequency Accuracy",
    "Enter desired bin width in Hz (e.g., 0.5, 0.125):",
    initialvalue=0.5,
    minvalue=0.0001,
    maxvalue=10.0,
    parent=tk_root
    )
else: #calculates bin_width based off of userStim
    bin_width = (userStim * 0.95) % 1 # checks margin needed for 95%
if bin_width is None:
    print("No bin width entered; exiting.")
    raise SystemExit

print("Bin Width: -----------------------V")
print(bin_width)


# margin is float, location is list, power is list
# e.g (pass 0.5 margin, list of peak frequencies, corresponding list of prominence)
def peakFinder(margin, location, power):#bin function change

    #creates bins "margin" wide starting at margin/2 till largest location
    binEdges = np.arange(margin/2, np.ceil(max(location)) + margin, margin)
    # grab the counts for each bin
    binGram, _ = np.histogram(location, bins=binEdges, weights=power)

    bestIndex = np.argmax(binGram) #strongest bin

    #midpoint between bin edges
    peak = (binEdges[bestIndex] + binEdges[bestIndex+1])/ 2.0
    return peak

# since the highest promFreq would be the stimrate if not for harmonics we can do:

calcStim = peakFinder(bin_width, top_freqs, top_proms)

print ("BELOW ARE TESTING VALUES --------------------------------------")
print("!!!!!!!Pairwise Diff Attempt: ")
print(str(peakFinder(bin_width, diffs_pairwise, weights_pairwise)))

print("**************Most Prominent Freq Attempt (likely harmonic):")
print(str(np.min(top_freqs)))

print("-------Expected Value:")
print(47.5)

if diffs is not None:
    estimatedStim = peakFinder(bin_width, diffs, weights)
    print("~~~~~~~~~~~~~~Adjacent differences attempt: ")
    print(estimatedStim)
    # check for harmonic in calcstim, this wouldnt work if calcstim is simply noise 
    possibleHarmonic = calcStim/estimatedStim % 1
    if possibleHarmonic < 0.05 or possibleHarmonic > 0.95: 
        calcStim = estimatedStim 

print("Final Calculated Stim : ")
print(calcStim)

if userStim is not None:
    print("User input reminder:")
    print(userStim)
    # check whether userStim is closer to 100% or 95%
    if abs(userStim - calcStim) > abs((userStim * 0.95) - calcStim) : 
        userStim *= 0.95 #if closer to 95% set to 95%
        print("95% user input stim is closer to calculated rate, user input will change")
    calcStim = userStim
    print("Final stim frequency post user stim comparison:")
    print(calcStim)

print("Stim frequency Finalized!!!!") 

#now calcStim is the final stim rate, treat it as such 

print("Time to start convolution!") #Section 2 starts here!


# Compute average signal across channels (ensure it's defined)
avg_signal = data.mean(axis=0)

# Initial rough detection of artifact starts via thresholding
# Define threshold as mean + 3*std of absolute signal
thresh_init = np.mean(np.abs(avg_signal)) + 3 * np.std(np.abs(avg_signal))
# Minimum distance between pulses in samples (~0.8 × period)
min_dist = int(0.8 * (sampleRate / calcStim))
# Find peaks in absolute average signal
peaks_init, _ = find_peaks(np.abs(avg_signal), height=thresh_init, distance=min_dist)
artifactStarts = peaks_init.tolist()

# --- Template Matching Section ---

# avg_signal: your mean signal across channels (1D array)
# artifactStarts: list/array of initial pulse sample indices
# sampleRate: sampling rate in Hz
# calcStim: your estimated stim frequency

# 1) Build the template by averaging snippets around detected pulses
window_ms = 5  # window half-width in milliseconds
window_samples = int(window_ms * sampleRate / 1000)
snippets = []

for idx in artifactStarts[:10]:  # use first 10 pulses
    start = max(0, idx - window_samples)
    end = min(len(avg_signal), idx + window_samples)
    snippet = avg_signal[start:end]
    # pad/truncate to uniform length
    if len(snippet) < 2*window_samples:
        snippet = np.pad(snippet, (0, 2*window_samples - len(snippet)), mode='constant')
    snippets.append(snippet)

template = np.mean(snippets, axis=0)

# 2) Matched filter: convolve with time-reversed template
mf_kernel = template[::-1]
mf_output = np.convolve(avg_signal, mf_kernel, mode='same')

# 3) Detect peaks in matched-filter output (80th percentile threshold, 0.5× period spacing)
period_samples = sampleRate / calcStim
peak_threshold = np.percentile(mf_output, 80)  # top 20% as threshold
distance = int(0.5 * period_samples)           # allow as close as half a period
peaks_mf, props_mf = find_peaks(
    mf_output,
    height=peak_threshold,
    distance=distance
)

# Use matched-filter peaks as artifact starts
artifactStarts = peaks_mf.tolist()
half_len = len(template) // 2
artifactEnds   = [min(len(avg_signal)-1, start + half_len) for start in artifactStarts]
print(f"Now detected {len(artifactStarts)} artifact pulses (template matching)")

# 4) Plot template and matched-filter response
#    Zoom into one pulse window around the first detected peak
fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)

# Template plot
t_template = (np.arange(len(template)) - len(template)//2) / sampleRate
ax_t.plot(t_template, template, color='C1')
ax_t.set_title("Derived Artifact Template")
ax_t.set_xlabel("Time (s)")
ax_t.set_ylabel("Amplitude")

# Matched-filter output
t_sig = np.arange(len(avg_signal)) / sampleRate
ax_m.plot(t_sig, mf_output, color='C2', label='Matched Filter Output')
ax_m.scatter(peaks_mf / sampleRate, mf_output[peaks_mf], color='red', marker='x', label='Detected Pulses')
# zoom around first pulse
t0 = artifactStarts[0] / sampleRate
ax_m.set_xlim(t0 - 0.05, t0 + 0.05)
ax_m.set_title("Matched-Filter Peaks (Zoomed)")
ax_m.set_xlabel("Time (s)")
ax_m.set_ylabel("Response")
ax_m.legend()

plt.show()


####### GPT VISSUALIZATION
# --- Visualization of template-matched artifact windows ---

# Automatically select the first channel if the hardcoded one is missing
if 'POL R CMa1-Ref' not in raw.ch_names:
    ch_name = raw.ch_names[0]
else:
    ch_name = 'POL R CMa1-Ref'
print(f"Plotting channel {ch_name} with artifact boundaries")
data_ch, times = raw.copy().pick(ch_name).get_data(return_times=True)
signal = data_ch[0]  # flatten to 1D

 # Define artifactEnds corresponding to each start using half-template length
half_len = len(template) // 2
artifactEnds = [min(len(avg_signal)-1, start + half_len) for start in artifactStarts]

# Convert artifact sample indices to time in seconds
t_starts = np.array(artifactStarts) / sampleRate
t_ends   = np.array(artifactEnds)   / sampleRate

plt.figure(figsize=(12, 4))
plt.plot(times, signal, label=ch_name, color='C0')

# Plot starts
plt.scatter(t_starts,
            signal[artifactStarts],
            c='green', marker='o', s=50,
            label='Artifact Starts')

# Plot ends
plt.scatter(t_ends,
            signal[artifactEnds],
            c='red', marker='x', s=50,
            label='Artifact Ends')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f"Channel {ch_name} with Template-Matched Artifact Windows")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


from scipy.interpolate import CubicSpline

def spline_stim_artifact(data, artifact_starts, artifact_ends):
    """
    Replace stimulation artifact segments in data via cubic spline interpolation.
    data: ndarray, shape (n_channels, n_samples)
    artifact_starts, artifact_ends: lists of sample indices
    """
    clean = data.copy()
    n_chan, _ = clean.shape
    for ch in range(n_chan):
        # skip channel if starts with NaN
        if np.isnan(clean[ch, 0]):
            continue
        for start, end in zip(artifact_starts, artifact_ends):
            # define points before and after artifact
            # ensure indices valid
            x = [start - 1, end + 1]
            y = clean[ch, x]
            # spline across artifact window
            cs = CubicSpline(x, y)
            xs = np.arange(start, end+1)
            clean[ch, xs] = cs(xs)
    return clean



# --- After plotting artifact boundaries, before spline interpolation ---

# 1) Debug print
print("First 10 artifact windows (samples):")
for s, e in zip(artifactStarts[:10], artifactEnds[:10]):
    print(f"  {s} → {e}")

# 2) Shade windows on raw signal
plt.figure(figsize=(12, 4))
plt.plot(times, signal, label='Raw Signal')
for s, e in zip(artifactStarts, artifactEnds):
    plt.axvspan(s / sampleRate, e / sampleRate, color='orange', alpha=0.3)
plt.title("Raw Signal with Artifact Windows Shaded")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# --- Now apply spline interpolation ---
print("Applying spline interpolation to remove artifacts...")
splineData = spline_stim_artifact(data, artifactStarts, artifactEnds)

# 3) Compare raw vs cleaned
cleaned = splineData[raw.ch_names.index(ch_name), :]
plt.figure(figsize=(12, 4))
plt.plot(times, signal,  label='Raw',   alpha=0.5)
plt.plot(times, cleaned, label='Cleaned', linewidth=1)
plt.title(f"Raw vs Cleaned Signal ({ch_name})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

raise SystemExit