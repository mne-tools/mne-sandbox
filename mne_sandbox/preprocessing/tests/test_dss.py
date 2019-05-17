# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose
import pytest

from mne import create_info, EpochsArray
from mne_sandbox.preprocessing import dss


def test_dss_args():
    """Test DSS error handling."""
    data1 = list()
    data2 = np.arange(6).reshape(2, 3)
    data3 = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    data4 = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    pytest.raises(TypeError, dss, data1)
    pytest.raises(ValueError, dss, data2)  # not enough dimensions
    pytest.raises(ValueError, dss, data4)  # too many dimensions
    pytest.raises(ValueError, dss, data3, data_thresh=2)  # invalid threshold


def test_dss():
    """Test DSS computations."""

    def rms(data):
        return np.sqrt(np.mean(data ** 2, axis=-1, keepdims=True))

    rand = np.random.RandomState(123)
    # parameters
    n_trials, n_times, n_channels, noise_dims, snr = [200, 1000, 16, 10, 0.1]
    # 1 Hz sine wave with silence before & after
    pad = np.zeros(n_times // 3)
    signal_nsamps = n_times - 2 * pad.size
    sine = np.sin(2 * np.pi * np.arange(signal_nsamps) / float(signal_nsamps))
    sine = np.r_[pad, sine, pad]
    signal = rand.randn(n_channels, 1) * sine[np.newaxis, :]
    # noise
    noise = np.einsum('hjk,ik->hij',
                      rand.randn(n_trials, n_times, noise_dims),
                      rand.randn(n_channels, noise_dims))
    # signal plus noise, in a reasonable range for EEG
    data = 4e-6 * (noise / rms(noise) + snr * signal / rms(signal))
    # perform DSS
    dss_mat, dss_data = dss(data, data_thresh=1e-3, bias_thresh=1e-3,
                            bias_max_components=n_channels - 1)
    # handle scaling and possible 180 degree phase difference
    dss_trial1_comp1 = dss_data[0, 0] / np.abs(dss_data[0, 0]).max()
    dss_trial1_comp1 *= np.sign(np.dot(dss_trial1_comp1, sine))
    assert_allclose(dss_trial1_comp1, sine, rtol=0, atol=0.2)
    # test handling of epochs objects
    sfreq = 1000
    first_samp = 150
    samps = np.arange(first_samp, first_samp + n_trials * n_times * 2,
                      n_times * 2)[:, np.newaxis]
    events = np.c_[samps, np.zeros_like(samps), np.ones_like(samps)]
    ch_names = ['EEG{0:03}'.format(n + 1) for n in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    epochs = EpochsArray(data, info=info, events=events, event_id={'fake': 1})
    dss_mat_epochs = dss(epochs, data_thresh=1e-3, bias_thresh=1e-3,
                         bias_max_components=n_channels - 1, return_data=False)
    # make sure we get the same answer when data is an epochs object
    dss_mat = dss_mat / dss_mat.max()
    dss_mat_epochs = dss_mat_epochs / dss_mat_epochs.max()
    assert_allclose(dss_mat, dss_mat_epochs)
