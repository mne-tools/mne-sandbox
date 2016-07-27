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
from mne_sandbox.connectivity import (simulate_pac_signal,
                                      PAC)
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
n_secs = 50.
n_events = 20
time = np.arange(0, n_secs, 1. / sfreq)
event_dur = 2.
event_times = np.linspace(1, n_secs - event_dur, n_events)
events = np.array(event_times) * sfreq
events = np.vstack([events, np.zeros_like(events), np.ones_like(events)])
events = events.astype(int).T

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

# Here we specify indices to calculate PAC in both directions
ixs = np.array([[0, 1],
                [1, 0]])

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
tmins = [0, .2]
tmaxs = [.2, .4]

# Create frequencies for our PAC estimation
freqs_phase = np.array([(i-.1, i+.1)
                        for i in np.arange(3, 12, 1)])
freqs_amp = np.array([(i-.1, i+.1)
                      for i in np.arange(f_amp-20, f_amp+20, 5)])
cycles_phase = 3.
cycles_amp = freqs_amp.mean(-1) / 10.

# We can then use a sklearn-style API to output a matrix of PAC values.
epochs = mne.Epochs(raw, events, tmin=-1, tmax=2)
pac = PAC(freqs_phase, freqs_amp, ixs, epochs.info['sfreq'], epochs.times,
          tmin=tmins, tmax=tmaxs, n_cycles_ph=cycles_phase,
          n_cycles_am=cycles_amp)

pac_vals = pac.fit_transform(epochs.get_data())

# Output is shape (n_epochs, n_t_win * n_f_win_ph * n_f_win_amp * n_ch_pairs)
fig, ax = plt.subplots()
ax.imshow(pac_vals, aspect='auto', cmap=plt.cm.viridis,
          interpolation='nearest')
ax.set_ylabel('Epoch')
ax.set_xlabel('PAC feature')
plt.show()
