# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose
import pytest

import mne
from mne_sandbox.connectivity import (phase_amplitude_coupling,
                                      phase_locked_amplitude,
                                      phase_binned_amplitude,
                                      simulate_pac_signal)
from sklearn.preprocessing import scale

np.random.seed(1337)
pac_func = 'ozkurt'
f_phase = 4
f_amp = 40
eptmin, eptmax = 1, 5
min_pac = .05
max_pac = .3

# First create PAC data

# Define parameters for our simulated signal
sfreq = 1000.
frac_pac = 1.  # This is the fraction of PAC to use
mag_ph = 4
mag_am = 1

# These are the times where PAC is active in our simulated signal
n_secs = 20.
time = np.arange(0, n_secs, 1. / sfreq)
event_times = np.arange(1, 18, 4)
events = (event_times * sfreq).astype(int)
event_dur = 2.

# Create a time mask that defines when PAC is active
msk_pac_times = np.zeros_like(time).astype(bool)
for i_time in event_times:
    msk_pac_times += mne.utils._time_mask(time, i_time, i_time + event_dur)
kws_sim = dict(mask_pac_times=msk_pac_times, snr_lo=10, snr_hi=10)
_, lo_pac, hi_pac = simulate_pac_signal(time, f_phase, f_amp, mag_ph,
                                        mag_am, frac_pac=frac_pac,
                                        **kws_sim)
_, lo_none, hi_none = simulate_pac_signal(time, f_phase, f_amp, mag_ph,
                                          mag_am, frac_pac=0, **kws_sim)

signal_a = lo_pac + hi_none
signal_b = lo_none + hi_pac

info = mne.create_info(['pac_hi', 'pac_lo'], sfreq, 'eeg')
raw = mne.io.RawArray([signal_a, signal_b], info)
events = np.vstack([events, np.zeros_like(events), np.ones_like(events)]).T
epochs = mne.Epochs(raw, events, tmin=eptmin, tmax=eptmax, baseline=None)


def test_phase_amplitude_coupling():
    """ Test phase amplitude coupling."""
    f_band_lo = [f_phase - 1, f_phase + 1]
    f_band_hi = [f_amp - 1, f_amp + 1]
    ixs_pac = [0, 1]
    ixs_no_pac = [1, 0]

    pytest.raises(ValueError,
                  phase_amplitude_coupling, epochs, f_band_lo,
                  f_band_hi, ixs_pac)

    # Testing Raw
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func)
    assert conn.mean() < min_pac
    assert conn.shape == (1, 1, 1, 1)

    # Testing Raw + multiple times
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func,
        tmin=event_times, tmax=event_times + event_dur)
    assert conn.mean() > max_pac
    assert conn.shape == (1, 1, 1, event_times.shape[0])
    # Difference in number of tmin / tmax
    pytest.raises(ValueError, phase_amplitude_coupling,
                  raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func,
                  tmin=event_times[1:], tmax=event_times + event_dur)

    # Testing Raw + multiple frequency pairs
    # Increasing n_cycles for better freq resolution
    f_band_lo_mult = [f_band_lo, (10, 12)]
    f_band_hi_mult = [f_band_hi, (14, 16)]
    conn, fbands = phase_amplitude_coupling(
        raw, f_band_lo_mult, f_band_hi_mult, ixs_pac, pac_func=pac_func,
        tmin=event_times, tmax=event_times + event_dur, n_cycles_ph=4,
        n_cycles_am=4)
    # Make sure shapes are right
    assert conn.shape[2] == 4
    assert len(fbands) == 4
    assert (conn[:, 0, 0, :].mean() > max_pac)  # pac freqs
    # Loosening the min value for frequency because of freq spillage
    assert (conn[:, 0, 1, :].mean() < .1)  # Non-pac freqs

    # Testing multiple n_cycles
    pytest.raises(ValueError, phase_amplitude_coupling, raw, f_band_lo_mult,
                  f_band_hi_mult, ixs_pac, pac_func=pac_func,
                  n_cycles_ph=[3, 4, 5], n_cycles_am=4)
    pytest.raises(ValueError, phase_amplitude_coupling, raw, f_band_lo_mult,
                  f_band_hi_mult, ixs_pac, pac_func=pac_func,
                  tmin=event_times, tmax=event_times + event_dur,
                  n_cycles_ph=[[3, 4, 5]], n_cycles_am=4)

    # Testing Raw + multiple PAC
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=['ozkurt', 'glm'])
    assert conn.shape[0] == 2

    # Mixing hi-freq phase and hi-freq amplitude metrics
    pytest.raises(ValueError, phase_amplitude_coupling,
                  raw, f_band_lo, f_band_hi, ixs_no_pac,
                  pac_func=['ozkurt', 'plv'])

    # Testing Raw + Epochs
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func, events=events,
        tmin=0, tmax=event_dur)
    assert (conn.mean() > max_pac)
    assert conn.shape == (events.shape[0], 1, 1, 1)

    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func,
        events=events, tmin=0, tmax=event_dur)
    assert (conn.mean() < min_pac)

    # Testing Raw + Epochs + concatenating epochs
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func, events=events,
        tmin=0, tmax=event_dur, concat_epochs=True)
    assert (conn.mean() > max_pac)
    assert conn.shape == (1, 1, 1, 1)

    # Testing Raw + Epochs + multiple times
    # First time window should have PAC, second window doesn't
    # Testing hi end at .3 because ozkurt seems to peak here
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func, events=events,
        tmin=[0, -1], tmax=[event_dur, -.5])
    assert (conn[..., 0].mean() > max_pac)
    assert (conn[..., 1].mean() < min_pac)
    assert conn.shape == (events.shape[0], 1, 1, 2)

    # Same times but non-pac ixs
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func,
        events=events, tmin=[0, -1], tmax=[event_dur, -.5])
    assert (conn[..., 0].mean() < min_pac)
    assert (conn[..., 1].mean() < min_pac)
    assert conn.shape == (events.shape[0], 1, 1, 2)

    # Check return data and scale func
    conn, _, data_phase, data_amp = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func,
        return_data=True, scale_amp_func=scale)
    # Make sure amp has been scaled
    assert (np.abs(data_amp.mean()) < 1e-7)
    assert (np.abs(data_amp.std() - 1) < 1e-7)
    # Make sure we have phases
    assert_allclose(data_phase.max(), np.pi, rtol=1e-2)
    assert_allclose(data_phase.min(), -np.pi, rtol=1e-2)

    # Check that arrays don't work
    pytest.raises(
        ValueError, phase_amplitude_coupling, raw._data, f_band_lo, f_band_hi,
        [0, 1], pac_func=pac_func)
    # Make sure ixs at least length 2
    pytest.raises(
        ValueError, phase_amplitude_coupling, raw, f_band_lo, f_band_hi,
        [0], pac_func=pac_func)
    # f-band only has 1 value
    pytest.raises(
        ValueError, phase_amplitude_coupling, raw, f_band_lo,
        [1], [0, 1], pac_func=pac_func)
    # Wrong pac func
    pytest.raises(
        ValueError, phase_amplitude_coupling, raw, f_band_lo, f_band_hi,
        [0, 1], pac_func='blah')


def test_phase_amplitude_viz_funcs():
    """Test helper functions for visualization"""
    freqs_ph = np.linspace(8, 12, 2)
    freqs_amp = np.linspace(40, 60, 5)
    ix_ph = 0
    ix_amp = 1

    # Phase locked viz
    amp, phase, times = phase_locked_amplitude(
        epochs, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert amp.shape[-1] == phase.shape[-1] == times.shape[-1]

    amp, phase, times = phase_locked_amplitude(
        raw, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert amp.shape[-1] == phase.shape[-1] == times.shape[-1]

    use_times = raw.times < 3
    amp, phase, times = phase_locked_amplitude(
        raw, freqs_ph, freqs_amp, ix_ph, ix_amp, mask_times=use_times,
        tmin=-.5, tmax=.5)
    assert amp.shape[-1] == phase.shape[-1] == times.shape[-1]

    # Phase binning
    amp_binned, bins = phase_binned_amplitude(epochs, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert (amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert (amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20,
                                              mask_times=use_times)
    assert (amp_binned.shape[0] == bins.shape[0] - 1)


def test_phase_amplitude_coupling_simulation():
    both, lo_none, hi_none = simulate_pac_signal(time, f_phase, f_amp, mag_ph,
                                                 mag_am, frac_pac=1.,
                                                 **kws_sim)
    # Shapes are correct
    assert both.shape == lo_none.shape == hi_none.shape
    assert time.shape[-1] == both.shape[-1]

    # Fracs outside of 0 to 1
    pytest.raises(ValueError, simulate_pac_signal, time, f_phase, f_amp,
                  mag_ph, mag_am, frac_pac=-.5, **kws_sim)
    pytest.raises(ValueError, simulate_pac_signal, time, f_phase, f_amp,
                  mag_ph, mag_am, frac_pac=1.5, **kws_sim)
    # Giving a band for frequencies
    pytest.raises(ValueError, simulate_pac_signal, time, [1, 2], f_amp, mag_ph,
                  mag_am, frac_pac=-.5, **kws_sim)
