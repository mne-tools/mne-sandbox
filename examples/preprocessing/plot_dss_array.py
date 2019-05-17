# -*- coding: utf-8 -*-
"""Denoising source separation applied to a NumPy array."""

# Authors: Daniel McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import numpy as np
from mne_sandbox.preprocessing import dss
from matplotlib import pyplot as plt


def rms(data):
    return np.sqrt(np.mean(data ** 2, axis=-1, keepdims=True))

snr = 0.1
noise_dims = 20
rand = np.random.RandomState(123)

# create synthetic data
n_trials = 200
n_times = 1000
n_channels = 32
pad = np.zeros(n_times // 3)
signal_nsamps = n_times - 2 * pad.size
sine = np.sin(2 * np.pi * np.arange(signal_nsamps) / float(signal_nsamps))
signal = rand.randn(n_channels, 1) * np.r_[pad, sine, pad][np.newaxis, :]
channel_noise = rand.randn(n_channels, noise_dims)
trial_noise = rand.randn(n_trials, n_times, noise_dims)
noise = np.einsum('ijk,lk->ilj', trial_noise, channel_noise)
data = 4e-6 * (noise / rms(noise) + snr * signal / rms(signal))

# perform DSS
dss_mat, dss_data = dss(data, data_thresh=1e-3, bias_thresh=1e-3)

# plot
fig, axs = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
plotdata = [signal.T, data[0].T, dss_data[:, 0].T]
linewidths = (1, 0.3, 0.4)
titles = ('synthetic signal with random weights for each channel',
          'one trial, all channels, after noise addition (SNR=0.1)',
          'First DSS component from each trial')
for ax, dat, lw, ti in zip(axs, plotdata, linewidths, titles):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(dat, linewidth=lw)
    ax.set_title(ti)
ax.set_xlabel('samples')
plt.tight_layout()
