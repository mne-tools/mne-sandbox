# -*- coding: utf-8 -*-
"""Denoising source separation applied to an Epochs object."""

import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

from mne import EpochsArray, create_info
from mne_sandbox.preprocessing import dss

rng = np.random.RandomState(0)
sfreq = 1000.
f = 4.  # sinusoid freq
n_channels, n_times, n_epochs = 10, 1000, 100
max_delay = 20
n_noises = 9
sinusoid = np.sin(2 * np.pi * f * np.arange(int(round(sfreq / f))) / sfreq)
mixing = rng.randn(n_channels, n_noises)

data = np.zeros((n_epochs, n_channels, n_times))
samps = rng.randint(200, 200 + max_delay, n_epochs)
data[np.arange(n_epochs), :, samps] = 1
data = lfilter(sinusoid, [1], data, axis=-1)
data += np.einsum(
    'ent,cn->ect', rng.randn(n_epochs, n_noises, n_times), mixing)
info = create_info(n_channels, sfreq, 'eeg')

epochs = EpochsArray(data, info, tmin=-0.2)
evoked = epochs.average()

# perform DSS
_, dss_data = dss(epochs)

# perform tsDSS
_, tsdss_data = dss(epochs, max_delay=max_delay)

# plot
fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
plotdata = [data.mean(1).T, dss_data[:, 0].T, tsdss_data[:, 0].T]
titles = ('raw data',
          'first DSS component',
          'first tsDSS component')
for ax, dat, ti in zip(axs, plotdata, titles):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(1e3 * evoked.times, dat)
    ax.set_title(ti)
ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()
