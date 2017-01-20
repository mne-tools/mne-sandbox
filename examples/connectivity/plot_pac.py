"""
==============================================================
Compute phase-amplitude coupling measures between signals
==============================================================
Simulate phase-amplitude coupling between two signals, and computes
several PAC metrics between them. Calculates PAC for all timepoints, as well
as for time-locked PAC responses.
References
----------
[1] Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
    Berger MS, Barbaro NM, Knight RT. "High gamma power is phase-locked to
    theta oscillations in human neocortex." Science. 2006.
[2] Tort ABL, Komorowski R, Eichenbaum H, Kopell N. Measuring phase-amplitude
    coupling between neuronal oscillations of different frequencies. Journal of
    Neurophysiology. 2010.
"""
# Author: Chris Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD (3-clause)
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne_sandbox.connectivity import (PAC, simulate_pac_signal,
                                      phase_amplitude_coupling)
import logging

print(__doc__)
np.random.seed(1337)
logger = logging.getLogger('mne')
logger.setLevel(50)

###############################################################################
# Phase-amplitude coupling (PAC) is a technique to determine if the
# amplitude of a high-frequency signal is locked to the phase
# of a low-frequency signal. The phase_amplitude_coupling function
# calculates PAC between pairs of signals for one or multiple
# time windows. In this example, we'll simulate two signals. One
# of the signals has an amplitude that is locked to the phase of
# the other. We'll calculate PAC for a number of time points, and
# in both directions to show how PAC responds.

# Define parameters for our simulated signal
sfreq = 1000.
f_phase = 5
f_amp = 40
frac_pac = .99  # This is the fraction of PAC to use
mag_ph = 4
mag_am = 1

# These are the times where PAC is active in our simulated signal
n_secs = 30.
time = np.arange(0, n_secs, 1. / sfreq)
event_times = np.arange(1, 28, 4)
event_dur = 2.

# Create a time mask that defines when PAC is active
msk_pac_times = np.zeros_like(time).astype(bool)
for i_time in event_times:
    msk_pac_times += mne.utils._time_mask(time, i_time, i_time + event_dur)

# Now simulate two signals. First, a low-frequency phase
# that modulates high-frequency amplitude
_, lo_pac, hi_pac = simulate_pac_signal(time, f_phase, f_amp, mag_ph, mag_am,
                                        frac_pac=frac_pac,
                                        mask_pac_times=msk_pac_times)

# Now two signals with no relationship between them
_, lo_none, hi_none = simulate_pac_signal(time, f_phase, f_amp, mag_ph,
                                          mag_am, frac_pac=0,
                                          mask_pac_times=msk_pac_times)

# Finally we'll mix them up.
# The low-frequency phase of signal A...
signal_a = lo_pac + hi_none
# Modulates the high-frequency amplitude of signal B. But not the reverse.
signal_b = lo_none + hi_pac


# To standardize the scales of each PAC metric
def normalize_pac(pac):
    return pac / np.mean(pac)


# We'll visualize these signals. A on the left, B on the right
# The top row is a combination of the middle and bottom row
labels = ['Combined Signal', 'Lo-Freq signal', 'Hi-freq signal']
data = [[signal_a, lo_none, hi_pac],
        [signal_b, lo_pac, hi_none]]
fig, axs = plt.subplots(3, 2, figsize=(10, 5))
for axcol, i_data in zip(axs.T, data):
    for ax, i_sig, i_label in zip(axcol, i_data, labels):
        ax.plot(time, i_sig)
        ax.set_title(i_label, fontsize=20)
_ = plt.setp(axs, xlim=[8, 12])
plt.tight_layout()

# Create a raw array from the simulated data
info = mne.create_info(['pac_hi', 'pac_lo'], sfreq, 'eeg')
raw = mne.io.RawArray([signal_a, signal_b], info)

# The PAC function needs a lower and upper bound for each frequency
f_phase_bound = (f_phase-.1, f_phase+.1)
f_amp_bound = (f_amp-2, f_amp+2)

# First we'll calculate PAC for the entire timeseries.
# We'll use a few PAC metrics to compare.
iter_pac_funcs = [['mi_tort', 'ozkurt'], ['plv']]
win_size = 1.  # In seconds
step_size = .1
pac_times = np.array(
    [(i, i + win_size)
     for i in np.arange(0, np.max(time) - win_size, step_size)])

# Here we specify indices to calculate PAC in both directions
ch_ixs = np.array([[0, 1],
                [1, 0]])
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
ylim = 0
all_pac = []
for pac_funcs in iter_pac_funcs:
    pac, pac_freqs = phase_amplitude_coupling(
        raw, (f_phase-.1, f_phase+.1), (f_amp-.5, f_amp+.5), ch_ixs,
        pac_func=pac_funcs, tmin=pac_times[:, 0], tmax=pac_times[:, 1],
        n_cycles_ph=3, n_cycles_am=3)
    pac = pac.squeeze()
    if len(pac_funcs) == 1:
        pac = pac[np.newaxis, ...]
    for i in range(pac.shape[0]):
        pac[i] = normalize_pac(pac[i])
    # Now plot timeseries of each
    for i_pac, i_name in zip(pac, pac_funcs):
        for i_pac_ix, ax in zip(i_pac.squeeze(), axs):
            ax.plot(pac_times.mean(-1), i_pac_ix, label=i_name)
    ylim = np.max(pac) if np.max(pac) > ylim else ylim  # For proper scale
axs[0].legend()
axs[0].set_title('PAC: low-freq A to high-freq B', fontsize=20)
axs[1].set_title('PAC: low-freq B to high-freq A', fontsize=20)
_ = plt.setp(axs, ylim=[0, ylim + .1])
plt.tight_layout()


# We can also calculate event-locked PAC
# by supplying a list of event indices
ev = np.array(event_times) * sfreq
ev = ev.astype(int)

pac_funcs = ['mi_tort', 'ozkurt']
colors = ['b', 'r']
win_size = 1.
ev_tmin = -2.
ev_tmax = 3.
pac_times = np.array([(i, i + win_size)
                      for i in np.arange(ev_tmin, ev_tmax, step_size)])
pac, pac_freqs = phase_amplitude_coupling(
    raw, (f_phase-.1, f_phase+.1), (f_amp-.5, f_amp+.5), ch_ixs,
    pac_func=pac_funcs, tmin=pac_times[:, 0], tmax=pac_times[:, 1],
    events=ev, concat_epochs=False)
pac = pac.squeeze()
for i in range(pac.shape[0]):
    pac[i] = normalize_pac(pac[i])

# This allows us to calculate the stability of PAC across epochs
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
for ii, (i_pac, i_pac_name, color) in enumerate(zip(pac, pac_funcs, colors)):
    for i_pac_mn, ax in zip(i_pac.mean(0), axs):
        ax.plot(pac_times[:, 0], i_pac_mn, color=color, label=i_pac_name)
        ax.axvline(0, ls='--', color='k', lw=1)

axs[0].legend()
axs[0].set_title('Time-locked PAC: Signal A to Signal B')
axs[1].set_title('Time-locked PAC: Signal B to Signal A')

plt.tight_layout()

# Finally, we can also calculate PAC with multiple frequencies
freqs_phase = np.array([(i-.1, i+.1)
                        for i in np.arange(3, 12, 1)])
freqs_amp = np.array([(i-.1, i+.1)
                      for i in np.arange(f_amp-20, f_amp+20, 5)])
# We can set a variable number of cycles per freq to control bandwidth
cycles_phase = 5.
cycles_amp = freqs_amp.mean(-1) / 10.
pac, pac_freqs = phase_amplitude_coupling(
    raw, freqs_phase, freqs_amp, ch_ixs,
    pac_func=['ozkurt'], tmin=.2, tmax=event_dur - .2,
    events=ev, concat_epochs=True, n_cycles_ph=cycles_phase,
    n_cycles_am=cycles_amp)
pac = pac.squeeze()
pac = normalize_pac(pac)

f, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, i_pac in zip(axs, pac):
    comod = i_pac.reshape([-1, len(freqs_amp)]).T
    ax.pcolormesh(freqs_phase[:, 0], freqs_amp[:, 0], comod,
                  vmin=0, vmax=np.max(pac) + .1)
_ = plt.setp(axs, xlim=[freqs_phase[:, 0].min(), freqs_phase[:, 0].max()],
             ylim=[freqs_amp[:, 0].min(), freqs_amp[:, 0].max()],
             xlabel='Frequency Phase (Hz)', ylabel='Frequency Amplitude (Hz)')
axs[0].set_title('Comodulogram: Signal A to Signal B')
axs[1].set_title('Comodulogram: Signal B to Signal A')
plt.tight_layout()

# We can also use a sklearn-style API to output a matrix of PAC values.
tmins = [0, .2, .4]
tmaxs = [.2, .4, .6]
events = np.array(event_times) * sfreq
events = np.vstack([events, np.zeros_like(events), np.ones_like(events)])
events = events.astype(int).T
epochs = mne.Epochs(raw, events, tmin=-1, tmax=2, baseline=None,
                    add_eeg_ref=False)

# A PAC estimator expects data in Epochs format.
pac = PAC(freqs_phase, freqs_amp, ch_ixs, epochs.times, epochs.info['sfreq'],
          tmin=tmins, tmax=tmaxs, n_cycles_ph=cycles_phase,
          n_cycles_am=cycles_amp, pac_func='ozkurt')
pac_vals = pac.transform(epochs.get_data())
freq_pairs = pac.freq_pairs_.mean(-1)

fig, axs = plt.subplots(1, 2)
for i_ch_pair in range(pac_vals.shape[1]):
    i_pac = pac_vals[:, i_ch_pair, ...].mean(-1)
    ax = axs[i_ch_pair]
    ax.imshow(i_pac, aspect='auto', cmap=plt.cm.viridis,
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_ylabel('Epoch')
    ax.set_xlabel('Frequency pair')
    xtick_ixs = np.arange(0, 72, 10)
    ax.set_xticks(xtick_ixs)
    ax.set_xticklabels(freq_pairs[xtick_ixs], rotation=45)
axs[0].set_title('Estimator PAC\n Signal A to Signal B')
axs[1].set_title('Estimator PAC\n Signal B to Signal A')
plt.show(block=True)
