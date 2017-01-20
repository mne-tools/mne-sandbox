# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import mne
from nose.tools import assert_true, assert_raises, assert_equal
from numpy.testing import assert_allclose, assert_array_equal
from mne_sandbox.connectivity import (phase_amplitude_coupling,
                                      phase_locked_amplitude,
                                      phase_binned_amplitude,
                                      simulate_pac_signal, PAC)
from mne_sandbox.connectivity.cfc import _phase_amplitude_coupling
from sklearn.preprocessing import scale

np.random.seed(1337)
pac_func = 'ozkurt'
f_phase = 4
f_amp = 40
eptmin, eptmax = 0, 5
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
event_times = np.arange(1, 14, 4)
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

info = mne.create_info(['pac_hi', 'pac_lo'], sfreq, 'ecog')
raw = mne.io.RawArray([signal_a, signal_b], info)
events = np.vstack([events, np.zeros_like(events), np.ones_like(events)]).T
epochs = mne.Epochs(raw, events, tmin=eptmin, tmax=eptmax, baseline=None)


def test_phase_amplitude_coupling():
    """ Test phase amplitude coupling."""
    f_band_lo = [f_phase - 1, f_phase + 1]
    f_band_hi = [f_amp - 1, f_amp + 1]
    ixs_pac = [0, 1]
    ixs_no_pac = [1, 0]

    assert_raises(ValueError, phase_amplitude_coupling, epochs, f_band_lo,
                  f_band_hi, ixs_pac)
    assert_raises(ValueError, _phase_amplitude_coupling, epochs.get_data()[0],
                  epochs.info['sfreq'], f_band_lo, f_band_hi, ixs_pac)

    # Testing Raw
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func)
    assert_true(conn.mean() < min_pac)
    assert_equal(conn.shape, (1, 1, 1, 1))
    # Make sure it also works w/ PLV for high-freq phase calculation
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func='plv')

    # Testing Raw + multiple times
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func,
        tmin=event_times, tmax=event_times + event_dur)
    assert_true(conn.mean() > max_pac)
    assert_equal(conn.shape, (1, 1, 1, event_times.shape[0]))
    # Difference in number of tmin / tmax
    assert_raises(ValueError, phase_amplitude_coupling,
                  raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func,
                  tmin=event_times[1:], tmax=event_times + event_dur)

    # Testing Raw + multiple frequency pairs
    # Increasing n_cycles for better freq resolution
    f_band_lo_mult = [f_band_lo, (10, 12)]
    f_band_hi_mult = [f_band_hi, (14, 16)]
    conn, fbands = phase_amplitude_coupling(
        raw, f_band_lo_mult, f_band_hi_mult, ixs_pac, pac_func=pac_func,
        tmin=event_times, tmax=event_times + event_dur, n_cycles_ph=4,
        n_cycles_amp=4)

    # Make sure shapes are right
    assert_equal(conn.shape[2], 4)
    assert_equal(len(fbands), 4)
    assert_true(conn[:, 0, 0, :].mean() > max_pac)  # pac freqs
    # Loosening the min value for frequency because of freq spillage
    assert_true(conn[:, 0, 1, :].mean() < .1)  # Non-pac freqs

    # Testing multiple n_cycles
    assert_raises(ValueError, phase_amplitude_coupling, raw, f_band_lo_mult,
                  f_band_hi_mult, ixs_pac, pac_func=pac_func,
                  n_cycles_ph=[3, 4, 5], n_cycles_amp=4)
    assert_raises(ValueError, phase_amplitude_coupling, raw, f_band_lo_mult,
                  f_band_hi_mult, ixs_pac, pac_func=pac_func,
                  tmin=event_times, tmax=event_times + event_dur,
                  n_cycles_ph=[[3, 4, 5]], n_cycles_amp=4)

    # Testing Raw + multiple PAC
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=['ozkurt', 'glm'])
    assert_equal(conn.shape[0], 2)

    # Mixing hi-freq phase and hi-freq amplitude metrics
    assert_raises(ValueError, phase_amplitude_coupling,
                  raw, f_band_lo, f_band_hi, ixs_no_pac,
                  pac_func=['ozkurt', 'plv'])

    # Testing Raw + Epochs
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func, events=events,
        tmin=0, tmax=event_dur)
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func,
        events=events[:, 0], tmin=0, tmax=event_dur)
    # tmin/tmax must both be given
    assert_raises(ValueError, phase_amplitude_coupling,
                  raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func,
                  events=events[:, 0], tmin=None, tmax=event_dur)
    # Incorrect n dims for events
    assert_raises(ValueError, phase_amplitude_coupling, raw, f_band_lo,
                  f_band_hi, ixs_pac, pac_func=pac_func, events=[events[:, 0]],
                  tmin=0, tmax=event_dur)
    # Incorrect n columns for events
    assert_raises(ValueError, phase_amplitude_coupling, raw, f_band_lo,
                  f_band_hi, ixs_pac, pac_func=pac_func, events=events[:, :2],
                  tmin=0, tmax=event_dur)
    # tmin < times.min()
    assert_raises(ValueError, phase_amplitude_coupling, raw, f_band_lo,
                  f_band_hi, ixs_pac, pac_func=pac_func, events=events[:, :2],
                  tmin=-9000, tmax=event_dur)
    assert_raises(ValueError, phase_amplitude_coupling, raw, f_band_lo,
                  f_band_hi, ixs_pac, pac_func=pac_func, events=events[:, :2],
                  tmin=0, tmax=9000)

    assert_true(conn.mean() > max_pac)
    assert_equal(conn.shape, (events.shape[0], 1, 1, 1))

    pac = PAC(f_band_lo, f_band_hi, [ixs_pac, ixs_no_pac], times=epochs.times,
              sfreq=epochs.info['sfreq'], tmin=0, tmax=event_dur,
              pac_func=pac_func)
    pac_est = pac.transform(epochs.get_data())
    assert_true(pac_est[:, 0, :].mean() > max_pac)
    assert_true(pac_est[:, 1, :].mean() < min_pac)

    assert_array_equal(pac.freq_pairs_[0], [f_band_lo, f_band_hi])
    assert_raises(ValueError, pac.transform, epochs.get_data()[0])

    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func,
        events=events, tmin=0, tmax=event_dur)
    assert_true(conn.mean() < min_pac)

    # Testing Raw + Epochs + concatenating epochs
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func, events=events,
        tmin=0, tmax=event_dur, concat_epochs=True)
    assert_true(conn.mean() > max_pac)
    assert_equal(conn.shape, (1, 1, 1, 1))

    # Testing Raw + Epochs + multiple times
    # First time window should have PAC, second window doesn't
    # Testing hi end at .3 because ozkurt seems to peak here
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_pac, pac_func=pac_func, events=events,
        tmin=[0, -1], tmax=[event_dur, -.5])
    assert_true(conn[..., 0].mean() > max_pac)
    assert_true(conn[..., 1].mean() < min_pac)
    assert_equal(conn.shape, (events.shape[0], 1, 1, 2))

    # Same times but non-pac ixs
    conn, _ = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func,
        events=events, tmin=[0, -1], tmax=[event_dur, -.5])
    assert_true(conn[..., 0].mean() < min_pac)
    assert_true(conn[..., 1].mean() < min_pac)
    assert_equal(conn.shape, (events.shape[0], 1, 1, 2))

    # Check return data and scale func
    conn, _, data_phase, data_amp = phase_amplitude_coupling(
        raw, f_band_lo, f_band_hi, ixs_no_pac, pac_func=pac_func,
        return_data=True, scale_amp_func=scale)
    # Make sure amp has been scaled
    assert_true(np.abs(data_amp.mean()) < 1e-7)
    assert_true(np.abs(data_amp.std() - 1) < 1e-7)
    # Make sure we have phases
    assert_allclose(data_phase.max(), np.pi, rtol=1e-2)
    assert_allclose(data_phase.min(), -np.pi, rtol=1e-2)

    # Check that arrays don't work
    assert_raises(
        ValueError, phase_amplitude_coupling, raw._data, f_band_lo, f_band_hi,
        [0, 1], pac_func=pac_func)
    # Make sure ixs at least length 2
    assert_raises(
        ValueError, phase_amplitude_coupling, raw, f_band_lo, f_band_hi,
        [0], pac_func=pac_func)
    # f-band only has 1 value
    assert_raises(
        ValueError, phase_amplitude_coupling, raw, f_band_lo,
        [1], [0, 1], pac_func=pac_func)
    # Wrong pac func
    assert_raises(
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
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    amp, phase, times = phase_locked_amplitude(
        raw, freqs_ph, freqs_amp, ix_ph, ix_amp)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    use_times = raw.times < 3
    amp, phase, times = phase_locked_amplitude(
        raw, freqs_ph, freqs_amp, ix_ph, ix_amp, mask_times=use_times,
        tmin=-.5, tmax=.5)
    assert_equal(amp.shape[-1], phase.shape[-1], times.shape[-1])

    # Phase binning
    amp_binned, bins = phase_binned_amplitude(epochs, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)

    amp_binned, bins = phase_binned_amplitude(raw, freqs_ph, freqs_amp,
                                              ix_ph, ix_amp, n_bins=20,
                                              mask_times=use_times)
    assert_true(amp_binned.shape[0] == bins.shape[0] - 1)


def test_phase_amplitude_coupling_simulation():
    both, lo_none, hi_none = simulate_pac_signal(time, f_phase, f_amp, mag_ph,
                                                 mag_am, frac_pac=1.,
                                                 **kws_sim)
    # Shapes are correct
    assert_equal(both.shape, lo_none.shape, hi_none.shape)
    assert_equal(time.shape[-1], both.shape[-1])

    # Fracs outside of 0 to 1
    assert_raises(ValueError, simulate_pac_signal, time, f_phase, f_amp,
                  mag_ph, mag_am, frac_pac=-.5, **kws_sim)
    assert_raises(ValueError, simulate_pac_signal, time, f_phase, f_amp,
                  mag_ph, mag_am, frac_pac=1.5, **kws_sim)
    # Giving a band for frequencies
    assert_raises(ValueError, simulate_pac_signal, time, [1, 2], f_amp, mag_ph,
                  mag_am, frac_pac=-.5, **kws_sim)


if __name__ == '__main__':
    test_phase_amplitude_coupling()
    test_phase_amplitude_viz_funcs()
    test_phase_amplitude_coupling_simulation()
