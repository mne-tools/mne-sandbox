import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.time_frequency.tfr import _compute_tfr
from mne.preprocessing import peak_finder
from mne.utils import ProgressBar, logger
from scipy.signal import hilbert
from itertools import product
import mne
import warnings

if not callable(peak_finder):
    peak_finder = peak_finder.peak_finder  # old MNE


# Supported PAC functions
_pac_funcs = ['plv', 'glm', 'mi_tort', 'mi_canolty', 'ozkurt', 'otc']
# Calculate the phase of the amplitude signal for these PAC funcs
_hi_phase_funcs = ['plv']


def phase_amplitude_coupling(inst, f_phase, f_amp, ixs, pac_func='ozkurt',
                             events=None, tmin=None, tmax=None,
                             n_cycles_ph=3, n_cycles_am=3,
                             scale_amp_func=None, return_data=False,
                             concat_epochs=False, n_jobs=1, verbose=None):
    """ Compute phase-amplitude coupling between pairs of signals using pacpy.

    Parameters
    ----------
    inst : an instance of Raw or Epochs
        The data used to calculate PAC.
    f_phase : array, dtype float, shape (n_bands_phase, 2,)
        The frequency ranges to use for the phase carrier. PAC will be
        calculated between n_bands_phase * n_bands_amp frequencies.
    f_amp : array, dtype float, shape (n_bands_amp, 2,)
        The frequency ranges to use for the phase-modulated amplitude.
        PAC will be calculated between n_bands_phase * n_bands_amp frequencies.
    ixs : array-like, shape (n_ch_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_ch_pairs of channels. Indices correspond to rows of `data`.
    pac_func : {'plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt'} |
               list of strings
        The function for estimating PAC. Corresponds to functions in
        `pacpy.pac`. Defaults to 'ozkurt'. If multiple frequency bands are used
        then `plv` cannot be calculated.
    events : array, shape (n_events, 3) | array, shape (n_events,) | None
        MNE events array. To be supplied if data is 2D and output should be
        split by events. In this case, `tmin` and `tmax` must be provided. If
        `ndim == 1`, it is assumed to be event indices, and all events will be
        grouped together.
    tmin : float | list of floats, shape (n_pac_windows,) | None
        If `events` is not provided, it is the start time to use in `inst`.
        If `events` is provided, it is the time (in seconds) to include before
        each event index. If a list of floats is given, then PAC is calculated
        for each pair of `tmin` and `tmax`. Defaults to `min(inst.times)`.
    tmax : float | list of floats, shape (n_pac_windows,) | None
        If `events` is not provided, it is the stop time to use in `inst`.
        If `events` is provided, it is the time (in seconds) to include after
        each event index. If a list of floats is given, then PAC is calculated
        for each pair of `tmin` and `tmax`. Defaults to `max(inst.n_times)`.
    n_cycles_ph : float, int | array of floats, shape (n_bands_phase,)
        The number of cycles to be included in the window for each band-pass
        filter for phase. Defaults to 3.
    n_cycles_am : float, int | array of floats, shape (n_bands_amp,)
        The number of cycles to be included in the window for each band-pass
        filter for amplitude. Defaults to 3.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., `sklearn.preprocessing.scale`.
        Defaults to no scaling.
    return_data : bool
        If False, output will be `[pac_out]`. If True, output will be,
        `[pac_out, phase_signal, amp_signal]`.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see `mne.verbose`).

    Returns
    -------
    pac_out : array, list of arrays, dtype float,
              shape([n_pac_funcs], n_epochs, n_channel_pairs,
                    n_freq_pairs, n_pac_windows).
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs. If multiple pac metrics are specified, there will be one
        array per metric in the output list. If n_pac_funcs is 1, then the
        first dimension will be dropped.
    [phase_signal] : array, shape (n_phase_signals, n_times,)
        Only returned if `return_data` is True. The phase timeseries of the
        phase signals (first column of `ixs`).
    [amp_signal] : array, shape (n_amp_signals, n_times,)
        Only returned if `return_data` is True. The amplitude timeseries of the
        amplitude signals (second column of `ixs`).

    References
    ----------
    [1] This function uses the PacPy module developed by the Voytek lab.
        https://github.com/voytekresearch/pacpy
    """
    if not isinstance(inst, BaseRaw):
        raise ValueError('Must supply Raw as input')
    sfreq = inst.info['sfreq']
    data = inst[:][0]
    pac = _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                                    pac_func=pac_func, events=events,
                                    tmin=tmin, tmax=tmax,
                                    n_cycles_ph=n_cycles_ph,
                                    n_cycles_am=n_cycles_am,
                                    scale_amp_func=scale_amp_func,
                                    return_data=return_data,
                                    concat_epochs=concat_epochs,
                                    n_jobs=n_jobs, verbose=verbose)
    # Collect the data properly
    if return_data is True:
        pac, freq_pac, data_ph, data_am = pac
        return pac, freq_pac, data_ph, data_am
    else:
        pac, freq_pac = pac
        return pac, freq_pac


def _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                              pac_func='ozkurt', events=None,
                              tmin=None, tmax=None, n_cycles_ph=3,
                              n_cycles_am=3, scale_amp_func=None,
                              return_data=False, concat_epochs=False,
                              n_jobs=1, verbose=None):
    """ Compute phase-amplitude coupling using pacpy.

    Parameters
    ----------
    data : array, shape ([n_epochs], n_channels, n_times)
        The data used to calculate PAC
    sfreq : float
        The sampling frequency of the data.
    f_phase : array, dtype float, shape (n_bands_phase, 2,)
        The frequency ranges to use for the phase carrier. PAC will be
        calculated between n_bands_phase * n_bands_amp frequencies.
    f_amp : array, dtype float, shape (n_bands_amp, 2,)
        The frequency ranges to use for the phase-modulated amplitude.
        PAC will be calculated between n_bands_phase * n_bands_amp frequencies.
    ixs : array-like, shape (n_ch_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_ch_pairs of channels. Indices correspond to rows of `data`.
    pac_func : {'plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt'} |
               list of strings
        The function for estimating PAC. Corresponds to functions in
        `pacpy.pac`. Defaults to 'ozkurt'. If multiple frequency bands are used
        then `plv` cannot be calculated.
    events : array, shape (n_events, 3) | array, shape (n_events,) | None
        MNE events array. To be supplied if data is 2D and output should be
        split by events. In this case, `tmin` and `tmax` must be provided. If
        `ndim == 1`, it is assumed to be event indices, and all events will be
        grouped together.
    tmin : float | list of floats, shape (n_pac_windows,) | None
        If `events` is not provided, it is the start time to use in `inst`.
        If `events` is provided, it is the time (in seconds) to include before
        each event index. If a list of floats is given, then PAC is calculated
        for each pair of `tmin` and `tmax`. Defaults to `min(inst.times)`.
    tmax : float | list of floats, shape (n_pac_windows,) | None
        If `events` is not provided, it is the stop time to use in `inst`.
        If `events` is provided, it is the time (in seconds) to include after
        each event index. If a list of floats is given, then PAC is calculated
        for each pair of `tmin` and `tmax`. Defaults to `max(inst.n_times)`.
    n_cycles_ph : float, int | array of floats, shape (n_bands_phase,)
        The number of cycles to be included in the window for each band-pass
        filter for phase. Defaults to 3.
    n_cycles_am : float, int | array of floats, shape (n_bands_amp,)
        The number of cycles to be included in the window for each band-pass
        filter for amplitude. Defaults to 3.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., `sklearn.preprocessing.scale`.
        Defaults to no scaling.
    return_data : bool
        If False, output will be `[pac_out]`. If True, output will be,
        `[pac_out, phase_signal, amp_signal]`.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see `mne.verbose`).

    Returns
    -------
    pac_out : array, list of arrays, dtype float,
              shape([n_pac_funcs], n_epochs, n_channel_pairs,
                    n_freq_pairs, n_pac_windows).
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs. If multiple pac metrics are specified, there will be one
        array per metric in the output list. If n_pac_funcs is 1, then the
        first dimension will be dropped.
    [phase_signal] : array, shape (n_phase_signals, n_times,)
        Only returned if `return_data` is True. The phase timeseries of the
        phase signals (first column of `ixs`).
    [amp_signal] : array, shape (n_amp_signals, n_times,)
        Only returned if `return_data` is True. The amplitude timeseries of the
        amplitude signals (second column of `ixs`).
    """
    from ..externals.pacpy import pac as ppac
    pac_func = np.atleast_1d(pac_func)
    for i_func in pac_func:
        if i_func not in _pac_funcs:
            raise ValueError("PAC function %s is not supported" % i_func)
    n_pac_funcs = pac_func.shape[0]
    ixs = np.array(ixs, ndmin=2)
    n_ch_pairs = ixs.shape[0]
    tmin = 0 if tmin is None else tmin
    tmin = np.atleast_1d(tmin)
    n_pac_windows = len(tmin)
    tmax = (data.shape[-1] - 1) / float(sfreq) if tmax is None else tmax
    tmax = np.atleast_1d(tmax)
    f_phase = np.atleast_2d(f_phase)
    f_amp = np.atleast_2d(f_amp)
    n_cycles_ph = np.atleast_1d(n_cycles_ph)
    n_cycles_am = np.atleast_1d(n_cycles_am)
    if n_cycles_ph.shape[0] == 1:
        n_cycles_ph = np.repeat(n_cycles_ph, f_phase.shape[0])
    if n_cycles_am.shape[0] == 1:
        n_cycles_am = np.repeat(n_cycles_am, f_amp.shape[0])

    if data.ndim != 2:
        raise ValueError('Data must be shape (n_channels, n_times)')
    if ixs.shape[1] != 2:
        raise ValueError('Indices must have have a 2nd dimension of length 2')
    if f_phase.shape[-1] != 2 or f_amp.shape[-1] != 2:
        raise ValueError('Frequencies must be specified w/ a low/hi tuple')
    if len(tmin) != len(tmax):
        raise ValueError('tmin and tmax have differing lengths')
    if any(i_f.shape[0] > 1 and 'plv' in pac_func for i_f in (f_amp, f_phase)):
        raise ValueError('If calculating PLV, must use a single pair of freqs')
    for icyc, i_f in zip([n_cycles_ph, n_cycles_am], [f_phase, f_amp]):
        if icyc.shape[0] != i_f.shape[0]:
            raise ValueError("n_cycles must match n_freq_bands")
        if icyc.ndim > 1:
            raise ValueError("n_cycles must be 1-d, not {}d".format(icyc.ndim))

    logger.info('Pre-filtering data and extracting phase/amplitude...')
    hi_phase = np.unique([i_func in _hi_phase_funcs for i_func in pac_func])
    if len(hi_phase) != 1:
        raise ValueError("Can't mix pac funcs that use both hi-freq phase/amp")
    hi_phase = bool(hi_phase[0])
    data_ph, data_am, ix_map_ph, ix_map_am = _pre_filter_ph_am(
        data, sfreq, ixs, f_phase, f_amp, hi_phase=hi_phase,
        scale_amp_func=scale_amp_func, n_cycles_ph=n_cycles_ph,
        n_cycles_am=n_cycles_am)

    # So we know how big the PAC output will be
    if events is None:
        n_epochs = 1
    elif concat_epochs is True:
        if events.ndim == 1:
            n_epochs = 1
        else:
            n_epochs = np.unique(events[:, -1]).shape[0]
    else:
        n_epochs = events.shape[0]

    # Iterate through each pair of frequencies
    ixs_freqs = product(range(data_ph.shape[1]), range(data_am.shape[1]))
    ixs_freqs = np.atleast_2d(list(ixs_freqs))

    freq_pac = np.array([[f_phase[ii], f_amp[jj]] for ii, jj in ixs_freqs])
    n_f_pairs = len(ixs_freqs)
    pac = np.zeros([n_pac_funcs, n_epochs, n_ch_pairs,
                    n_f_pairs, n_pac_windows])
    for i_f_pair, (ix_f_ph, ix_f_am) in enumerate(ixs_freqs):
        # Second dimension is frequency
        i_f_data_ph = data_ph[:, ix_f_ph, ...]
        i_f_data_am = data_am[:, ix_f_am, ...]

        # Redefine indices to match the new data arrays
        ixs_new = [(ix_map_ph[i], ix_map_am[j]) for i, j in ixs]
        i_f_data_ph = mne.io.RawArray(
            i_f_data_ph, mne.create_info(i_f_data_ph.shape[0], sfreq))
        i_f_data_am = mne.io.RawArray(
            i_f_data_am, mne.create_info(i_f_data_am.shape[0], sfreq))

        # Turn into Epochs if we have defined events
        if events is not None:
            i_f_data_ph = _raw_to_epochs_mne(i_f_data_ph, events, tmin, tmax)
            i_f_data_am = _raw_to_epochs_mne(i_f_data_am, events, tmin, tmax)

        # Data is either Raw or Epochs
        pbar = ProgressBar(n_epochs)
        for itime, (i_tmin, i_tmax) in enumerate(zip(tmin, tmax)):
            # Pull times of interest
            with warnings.catch_warnings():  # To suppress a depracation
                warnings.simplefilter("ignore")
                # Not sure how to do this w/o copying
                i_t_data_am = i_f_data_am.copy().crop(i_tmin, i_tmax)
                i_t_data_ph = i_f_data_ph.copy().crop(i_tmin, i_tmax)

            if concat_epochs is True:
                # Iterate through each event type and hstack
                con_data_ph = []
                con_data_am = []
                for i_ev in i_t_data_am.event_id.keys():
                    con_data_ph.append(np.hstack(i_t_data_ph[i_ev]._data))
                    con_data_am.append(np.hstack(i_t_data_am[i_ev]._data))
                i_t_data_ph = np.vstack(con_data_ph)
                i_t_data_am = np.vstack(con_data_am)
            else:
                # Just pull all epochs separately
                i_t_data_ph = i_t_data_ph._data
                i_t_data_am = i_t_data_am._data
            # Now make sure that inputs to the loop are ep x chan x time
            if i_t_data_am.ndim == 2:
                i_t_data_ph = i_t_data_ph[np.newaxis, ...]
                i_t_data_am = i_t_data_am[np.newaxis, ...]
            # Loop through epochs (or epoch grps), each index pair, and funcs
            data_iter = zip(i_t_data_ph, i_t_data_am)
            for iep, (ep_ph, ep_am) in enumerate(data_iter):
                for iix, (i_ix_ph, i_ix_am) in enumerate(ixs_new):
                    for ix_func, i_pac_func in enumerate(pac_func):
                        func = getattr(ppac, i_pac_func)
                        pac[ix_func, iep, iix, i_f_pair, itime] = func(
                            ep_ph[i_ix_ph], ep_am[i_ix_am],
                            f_phase, f_amp, filterfn=False)
            pbar.update_with_increment_value(1)
    if pac.shape[0] == 1:
        pac = pac[0]
    if return_data:
        return pac, freq_pac, data_ph, data_am
    else:
        return pac, freq_pac


def _raw_to_epochs_mne(raw, events, tmin, tmax):
    """Convert Raw data to Epochs w/ some time checks."""
    events = np.atleast_1d(events)
    if events.ndim == 1:
        events = np.vstack([events, np.zeros_like(events),
                            np.ones_like(events)]).T
    if events.ndim != 2:
        raise ValueError('events have incorrect number of dimensions')
    if events.shape[-1] != 3:
        raise ValueError('events have incorrect number of columns')
    # Convert to Epochs using the event times
    tmin_all = np.min(tmin)
    tmax_all = np.max(tmax) + (1. / raw.info['sfreq'])
    return mne.Epochs(raw, events, tmin=tmin_all, tmax=tmax_all, preload=True,
                      baseline=None)


def _pre_filter_ph_am(data, sfreq, ixs, f_ph, f_am, n_cycles_ph=3,
                      n_cycles_am=3, hi_phase=False, scale_amp_func=None,
                      kws_filt=None):
    """Filter for phase/amp only once for each channel."""
    from ..externals.pacpy.pac import _range_sanity

    kws_filt = dict() if kws_filt is None else kws_filt
    ix_ph = np.atleast_1d(np.unique(ixs[:, 0]))
    ix_am = np.atleast_1d(np.unique(ixs[:, 1]))
    n_unique_am = ix_am.shape[0]

    # Filter for lo-freq phase
    for i_f_ph in f_ph:
        _range_sanity(i_f_ph, f_am[0])
    for i_f_am in f_am:
        _range_sanity(f_ph[0], i_f_am)

    # Output will be (n_chan, n_freqs, n_times). Operates in place.
    data_ph = data[ix_ph, :]
    out_ph = _filter_and_hilbert(data_ph, sfreq, f_ph, n_cycles_ph)
    out_ph = np.angle(out_ph)

    data_am = data[ix_am, :]
    out_am = _filter_and_hilbert(data_am, sfreq, f_am, n_cycles_am)
    out_am = np.abs(out_am)
    if hi_phase is True:
        # We assume f_ph has len(1), multiple freqs not supported w/ plv
        out_am = _filter_and_hilbert(out_am, sfreq, f_ph, n_cycles_ph,
                                     inplace=True)
        out_am = np.angle(out_am)

    # New index mapping for unique channels
    ix_map_ph = dict((ix, i) for i, ix in enumerate(ix_ph))
    ix_map_am = dict((ix, i) for i, ix in enumerate(ix_am))

    if scale_amp_func is not None:
        for ii in range(n_unique_am):
            out_am[ii] = scale_amp_func(out_am[ii], axis=-1)
    return out_ph, out_am, ix_map_ph, ix_map_am


def _raw_to_epochs_array(x, sfreq, events, tmin, tmax):
    """Aux function to create epochs from a 2D array"""
    if events.ndim != 1:
        raise ValueError('events must be 1D')
    if events.dtype != int:
        raise ValueError('events must be of dtype int')

    # Check that events won't be cut off
    n_times = x.shape[-1]
    min_ix = 0 - sfreq * tmin
    max_ix = n_times - sfreq * tmax
    msk_keep = np.logical_and(events > min_ix, events < max_ix)

    if not all(msk_keep):
        logger.info('Some event windows extend beyond data limits,'
                    ' and will be cut off...')
        events = events[msk_keep]

    # Pull events from the raw data
    epochs = []
    for ix in events:
        ix_min, ix_max = [ix + int(i_tlim * sfreq)
                          for i_tlim in [tmin, tmax]]
        epochs.append(x[np.newaxis, :, ix_min:ix_max])
    epochs = np.concatenate(epochs, axis=0)
    times = np.arange(epochs.shape[-1]) / float(sfreq) + tmin
    return epochs, times, msk_keep


def _filter_and_hilbert(data, sfreq, frequencies, n_cycles, inplace=False):
    if inplace is True:
        # Assume data is (n_chan, n_freqs, n_times)
        n_channels, n_freqs, n_times = data.shape
        out = data
    else:
        n_freqs = frequencies.shape[0]
        n_channels, n_times = data.shape
        data = data[:, np.newaxis, :]  # To ensure shapes are always consistent
        out = np.zeros([n_channels, n_freqs, n_times])
    # Filtering in place
    for ii in range(n_channels):
        for jj, n_cyc in zip(range(n_freqs), n_cycles):
            out[ii, jj] = _band_pass_pac(data[ii, 0], frequencies[jj], sfreq,
                                         n_cycles=n_cyc)
    n_hil = int(2 ** np.ceil(np.log2(n_times)))
    out = hilbert(out, N=n_hil)[..., :n_times]
    return out


def phase_locked_amplitude(inst, freqs_phase, freqs_amp, ix_ph, ix_amp,
                           tmin=-.5, tmax=.5, mask_times=None):
    """Calculate the average amplitude of a signal at a phase of another.

    Parameters
    ----------
    inst : instance of mne.Epochs | mne.io.Raw
        The data to be used in phase locking computation.
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation.
    ix_ph : int
        The index of the signal to be used for phase calculation.
    ix_amp : int
        The index of the signal to be used for amplitude calculation.
    tmin : float
        The time to include before each phase peak.
    tmax : float
        The time to include after each phase peak.
    mask_times : np.array, dtype bool, shape (inst.n_times,)
        If `inst` is an instance of Raw, this will only include times contained
        in `mask_times`. Defaults to using all times.

    Returns
    -------
    data_am : np.array
        The mean amplitude values for the frequencies specified in `freqs_amp`,
        time-locked to peaks of the low-frequency phase.
    data_ph : np.array
        The mean low-frequency signal, phase-locked to low-frequency phase
        peaks.
    times : np.array
        The times before / after each phase peak.
    """
    sfreq = inst.info['sfreq']
    # Pull the amplitudes/phases using Morlet
    data_ph, data_am = _pull_data(inst, ix_ph, ix_amp)
    angle_ph, band_ph, amp = _extract_phase_and_amp(
        data_ph, data_am, sfreq, freqs_phase, freqs_amp)

    angle_ph = angle_ph.mean(0)  # Mean across freq bands
    band_ph = band_ph.mean(0)

    # Find peaks in the phase for time-locking
    phase_peaks, vals = peak_finder(angle_ph)
    ixmin, ixmax = [t * sfreq for t in [tmin, tmax]]
    # Remove peaks w/o buffer
    phase_peaks = phase_peaks[(phase_peaks > np.abs(ixmin)) *
                              (phase_peaks < len(angle_ph) - ixmax)]

    if mask_times is not None:
        # Set datapoints outside out times to nan so we can drop later
        if len(mask_times) != angle_ph.shape[-1]:
            raise ValueError('mask_times must be == in length to data')
        band_ph[..., ~mask_times] = np.nan

    data_ph, times, msk_window = _raw_to_epochs_array(
        band_ph[np.newaxis, :], sfreq, phase_peaks, tmin, tmax)
    data_am, times, msk_window = _raw_to_epochs_array(
        amp, sfreq, phase_peaks, tmin, tmax)
    data_ph = data_ph.squeeze()
    data_am = data_am.squeeze()

    # Drop any peak events where there was a nan
    keep_rows = np.where(~np.isnan(data_ph).any(-1))[0]
    data_ph = data_ph[keep_rows, ...]
    data_am = data_am[keep_rows, ...]

    # Average across phase peak events
    data_am = data_am.mean(0)
    data_ph = data_ph.mean(0)
    return data_am, data_ph, times


def phase_binned_amplitude(inst, freqs_phase, freqs_amp,
                           ix_ph, ix_amp, n_bins=20, mask_times=None):
    """Calculate amplitude of one signal in sub-ranges of phase for another.

    Parameters
    ----------
    inst : instance of mne.Epochs | mne.io.Raw
        The data to be used in phase locking computation.
    freqs_phase : np.array, shape (n_freqs_phase,)
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array, shape (n_freqs_amp,)
        The frequencies to use in amplitude calculation. The amplitude
        of each frequency will be averaged together.
    ix_ph : int
        The index of the signal to be used for phase calculation.
    ix_amp : int
        The index of the signal to be used for amplitude calculation.
    n_bins : int
        The number of bins to use when grouping amplitudes. Each bin will
        have size `(2 * np.pi) / n_bins`.
    mask_times : np.array, dtype bool, shape (inst.n_times,)
        Remove timepoints where `mask_times` is False.
        Defaults to using all times.

    Returns
    -------
    amp_binned : np.array, shape (n_bins,)
        The mean amplitude of freqs_amp at each phase bin
    bins_phase : np.array, shape (n_bins + 1,)
        The bins used in the calculation. There is one extra bin because
        bins represent the left/right edges of each bin, not the center value.
    """
    sfreq = inst.info['sfreq']
    # Pull the amplitudes/phases using Morlet
    data_ph, data_am = _pull_data(inst, ix_ph, ix_amp)
    angle_ph, band_ph, amp = _extract_phase_and_amp(
        data_ph, data_am, sfreq, freqs_phase, freqs_amp)
    angle_ph = angle_ph.mean(0)  # Mean across freq bands
    if mask_times is not None:
        # Only keep times we want
        if len(mask_times) != amp.shape[-1]:
            raise ValueError('mask_times must be equal in length to data')
        angle_ph, band_ph, amp = [i[..., mask_times]
                                  for i in [angle_ph, band_ph, amp]]

    # Bin our phases and extract amplitudes based on bins
    bins_phase = np.linspace(-np.pi, np.pi, n_bins)
    bins_phase_ixs = np.digitize(angle_ph, bins_phase)
    unique_bins = np.unique(bins_phase_ixs)
    amp_binned = [np.mean(amp[:, bins_phase_ixs == i], axis=1)
                  for i in unique_bins]
    amp_binned = np.vstack(amp_binned).mean(1)

    return amp_binned, bins_phase


# For the viz functions
def _extract_phase_and_amp(data_ph, data_am, sfreq, freqs_phase,
                           freqs_amp, scale=True):
    """Extract the phase and amplitude of two signals for PAC viz.
    data should be shape (n_epochs, n_times)"""
    from sklearn.preprocessing import scale

    # Morlet transform to get complex representation
    band_ph = _compute_tfr([data_ph], freqs_phase, sfreq, method='morlet')[0]
    band_amp = _compute_tfr([data_ph], freqs_amp, sfreq, method='morlet')[0]

    # Calculate the phase/amplitude of relevant signals across epochs
    band_ph_stacked = np.hstack(np.real(band_ph))
    angle_ph = np.hstack(np.angle(band_ph))
    amp = np.hstack(np.abs(band_amp) ** 2)

    # Scale the amplitude for viz so low freqs don't dominate highs
    if scale is True:
        amp = scale(amp, axis=1)
    return angle_ph, band_ph_stacked, amp


def _pull_data(inst, ix_ph, ix_amp, events=None, tmin=None, tmax=None):
    """Pull data from either Base or Epochs instances"""
    if isinstance(inst, BaseEpochs):
        data_ph = inst.get_data()[:, ix_ph, :]
        data_am = inst.get_data()[:, ix_amp, :]
    elif isinstance(inst, BaseRaw):
        data = inst[[ix_ph, ix_amp], :][0].squeeze()
        data_ph, data_am = [i[np.newaxis, ...] for i in data]
    return data_ph, data_am


def _band_pass_pac(x, f_range, sfreq=1000, n_cycles=3):
    """
    Band-pass filter a signal using PacPy for PAC coupling.

    This is a version of the firf function in PacPy, minux edge removal.
    It's docstring is below
    ----
    Filter signal with an FIR filter
    *Like fir1 in MATLAB

    x : array-like, 1d
        Time series to filter.
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter.
    sfreq : float, Hz
        Sampling rate.
    n_cycles : float
        Length of the filter in terms of the number of cycles
        of the oscillation whose frequency is the low cutoff of the
        bandpass filter.

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series.
    """
    from ..externals.pacpy.filt import firwin, filtfilt

    if n_cycles <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    nyq = np.float(sfreq / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    n_taps = int(np.floor(n_cycles * sfreq / f_range[0]))
    if len(x) < n_taps:
        raise RuntimeError(
            'Length of filter is longer than data. '
            'Provide more data or a shorter filter.')

    # Perform filtering
    taps = firwin(n_taps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    return x_filt
