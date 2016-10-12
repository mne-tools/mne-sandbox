import numpy as np


def plot_phase_locked_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, mask_times=None,
                                normalize=True,
                                tmin=-.5, tmax=.5, return_data=False,
                                amp_kwargs=None, ph_kwargs=None):
    """Make a phase-locked amplitude plot.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    normalize : bool
        Whether amplitudes are normalized before averaging together. Helps
        if some frequencies have a larger mean amplitude than others.
    tmin : float
        The time to include before each phase peak
    tmax : float
        The time to include after each phase peak
    return_data : bool
        If True, the amplitude/frequency data will be returned
    amp_kwargs : dict
        kwargs to be passed to pcolormesh for amplitudes
    ph_kwargs : dict
        kwargs to be passed to the line plot for phase

    Returns
    -------
    axs : array of matplotlib axes
        The axes used for plotting.
    """
    import matplotlib.pyplot as plt
    from .cfc import phase_locked_amplitude
    from sklearn.preprocessing import scale
    amp_kwargs = dict() if amp_kwargs is None else amp_kwargs
    ph_kwargs = dict() if ph_kwargs is None else ph_kwargs

    # Handle kwargs defaults
    if 'cmap' not in amp_kwargs.keys():
        amp_kwargs['cmap'] = plt.cm.RdBu_r

    data_am, data_ph, times = phase_locked_amplitude(
        epochs, freqs_phase, freqs_amp,
        ix_ph, ix_amp, tmin=tmin, tmax=tmax, mask_times=mask_times)

    if normalize is True:
        # Scale within freqs across time
        data_am = scale(data_am, axis=-1)

    # Plotting
    f, axs = plt.subplots(2, 1)
    ax = axs[0]
    ax.pcolormesh(times, freqs_amp, data_am, **amp_kwargs)

    ax = axs[1]
    ax.plot(times, data_ph, **ph_kwargs)

    plt.setp(axs, xlim=[times[0], times[-1]])
    ylim = np.max(np.abs(ax.get_ylim()))
    plt.setp(ax, ylim=[-ylim, ylim])
    if return_data is True:
        return ax, data_am, data_ph
    else:
        return ax


def plot_phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                ix_ph, ix_amp, normalize=True,
                                n_bins=20, return_data=False,
                                mask_times=None, ax=None,
                                **kwargs):
    """Make a circular phase-binned amplitude plot.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to be used in phase locking computation
    freqs_phase : np.array
        The frequencies to use in phase calculation. The phase of each
        frequency will be averaged together.
    freqs_amp : np.array
        The frequencies to use in amplitude calculation. The amplitude
        of each frequency will be averaged together.
    ix_ph : int
        The index of the signal to be used for phase calculation
    ix_amp : int
        The index of the signal to be used for amplitude calculation
    normalize : bool
        Whether amplitudes are normalized before averaging together. Helps
        if some frequencies have a larger mean amplitude than others.
    n_bins : int
        The number of bins to use when grouping amplitudes. Each bin will
        have size (2 * np.pi) / n_bins.
    return_data : bool
        If True, the amplitude/frequency data will be returned
    ax : matplotlib axis | None
        If not None, plotting functions will be called on this object
    kwargs : dict
        kwargs to be passed to plt.bar

    Returns
    -------
    ax : matplotlib axis
        The axis used for plotting.
    """
    import matplotlib.pyplot as plt
    from .cfc import phase_binned_amplitude
    from sklearn.preprocessing import RobustScaler
    amps, bins = phase_binned_amplitude(epochs, freqs_phase, freqs_amp,
                                        ix_ph, ix_amp, n_bins=n_bins,
                                        mask_times=mask_times)
    if normalize is True:
        amps = RobustScaler().fit_transform(amps[:, np.newaxis])
    if ax is None:
        plt.figure()
        ax = plt.subplot(111, polar=True)
    bins_plt = bins[:-1]  # Because there is 1 more bins than amps
    width = 2 * np.pi / len(bins_plt)
    ax.bar(bins_plt + np.pi, amps, color='r', width=width)
    if return_data is True:
        return ax, amps, bins
    else:
        return ax
