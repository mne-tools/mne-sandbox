# -*- coding: utf-8 -*-
# Authors: Laura Gwilliams <laura.gwilliams@nyu.edu>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Alex Barachant <alexandre.barachant@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from mne.filter import filter_data


class TimeFrequencyCSP(object):
    u"""Decode MEG data in time-frequency space using a rolling covariance
    matrix.

    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.

    Parameters
    ----------
    tmin : float
        Time in seconds. Lower bound of the time over which to estimate
        signals.
    tmax : float
        Times in seconds. Upper bound of the time over which to estimate
        signals.
    freqs : array of floats
        Frequency values over which to compute time-frequency decomposition.
    estimator : object
        Scikit-learn classifier object.
    sfreq : float
        Sample frequency of data.
    n_cycles : float, defaults to 7.
        Number of cycles that each frequency should complete in forming the
        sliding temporal window.
        The time-window length is T = n_cycles / freq.
    scorer : object | None | str
        scikit-learn Scorer instance or str type indicating the name of the
        scorer such as ``accuracy``, ``roc_auc``. If None, set to ``accuracy``.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    ``filters_`` : ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    ``patterns_`` : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    ``mean_`` : ndarray, shape (n_components,)
        If fit, the mean squared power for each component.
    ``std_`` : ndarray, shape (n_components,)
        If fit, the std squared power for each component.
    """

    def __init__(self, tmin, tmax, freqs, estimator, sfreq, n_cycles=7.,
                 scorer=None, n_jobs=1):
        """Init of TimeFrequencyCSP."""

        # check frequency list
        if len(freqs) < 2:
            raise ValueError('freqs must contain more than one value.')
        self.freqs = freqs

        # check sfreq
        if not isinstance(sfreq, (float, int)):
            raise ValueError('sfreq must be a float or an int, got %s '
                             'instead.' % type(sfreq))
        self.sfreq = float(sfreq)

        # check n_cycles
        if not isinstance(n_cycles, (float, int)):
            raise ValueError('sfreq must be a float or an int, got %s '
                             'instead.' % type(n_cycles))
        self.n_cycles = float(n_cycles)

        # TODO:add test that the number of cycles is appropriate for epochs

        self.estimator = estimator
        self.scorer = scorer
        self.n_jobs = n_jobs

        # Assemble list of frequency range tuples
        self._freq_ranges = zip(self.freqs[:-1], freqs[1:])

        # Infer window spacing from the max freq and n cycles to avoid gaps
        self._window_spacing = (n_cycles / np.max(freqs) / 2.)
        self._centered_w_times = np.arange(tmin + self._window_spacing,
                                           tmax - self._window_spacing,
                                           self._window_spacing)[1:]  # noqa
        self._n_windows = len(self._centered_w_times)

    def _transform(self, epochs, y, w_tmin, w_tmax, method='', pos=[]):
        """
        Assumes data for one frequency band for one time window and fits CSP.
        """
        from sklearn import clone

        # Crop data into time-window of interest
        Xt = epochs.copy().crop(w_tmin, w_tmax).get_data()

        # call fit or predict depending on calling function
        if method == 'fit':
            self.estimator = clone(self.estimator)
            self._estimators[pos] = (self.estimator.fit(Xt, y))
            return self

        elif method == 'predict':
            self.y_pred = self._estimators[pos].predict(Xt)

        elif method == 'score':
            self.y_pred = self._estimators[pos].predict(Xt)
            return self._estimators[pos].score(Xt, y)

    def fit(self, epochs, y=None):
        """Train a classifier on each specified frequency and time slice.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        y : list or ndarray of int, shape (n_samples,) or None, optional
            To-be-fitted model values. If None, y = epochs.events[:, 2].

        Returns
        -------
        self : TimeFrequencyCSP
            Returns fitted TimeFrequencyCSP object.
        """
        n_freq = len(self._freq_ranges)
        n_window = len(self._centered_w_times)
        self._scores = np.zeros([n_freq, n_window])
        self._estimators = np.empty([n_freq, n_window], dtype=object)

        if y is None:
            y = epochs.events[:, 2]

        # loop through each frequency range
        for freq_ii, (fmin, fmax) in enumerate(self._freq_ranges):

            # Infer window size based on the frequency being used
            w_size = self.n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # filter the data at the desired frequency
            epochs_bp = epochs.copy()
            epochs_bp._data = filter_data(epochs_bp.get_data(), self.sfreq,
                                          fmin, fmax, verbose='CRITICAL')

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(self._centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # extract data for this time window
                self._transform(epochs_bp, y, w_tmin, w_tmax, method='fit',
                                pos=(freq_ii, t))
        pass

    def predict(self, epochs):
        """Test each classifier on each specified testing frequency and time
        slice.

        .. note::
            This function sets the ``y_pred_`` and ``test_times_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See
            predict_mode parameter.

        Returns
        -------
        y_pred :
            The single-trial predictions at each time window.
        """
        # loop through each frequency range
        for freq_ii, (fmin, fmax) in enumerate(self._freq_ranges):

            # Infer window size based on the frequency being used
            w_size = self.n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # filter the data at the desired frequency
            epochs_bp = epochs.copy()
            epochs_bp._data = filter_data(epochs_bp.get_data(), self.sfreq,
                                          fmin, fmax, verbose='CRITICAL')

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(self._centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # extract data for this time window
                self._transform(epochs_bp, None, w_tmin, w_tmax,
                                method='predict', pos=(freq_ii, t))
        pass

    def score(self, epochs, y):
        """Score Epochs.

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Calls ``predict()``.

        .. note::
            The function updates the ``scorer_``, ``scores_``, and
            ``y_true_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs | None, optional
            The epochs. Can be similar to fitted epochs or not.
            If None, it needs to rely on the predictions ``y_pred_``
            generated with ``predict()``.
        y : list | ndarray, shape (n_epochs,) | None, optional
            True values to be compared with the predictions ``y_pred_``
            generated with ``predict()`` via ``scorer_``.

        Returns
        -------
        scores : list of float, shape (n_times,)
            The scores estimated by ``scorer_`` at each time sample (e.g. mean
            accuracy of ``predict(X)``).
        """
        if y is None:
            y = epochs.events[:, 2]

        # loop through each frequency range
        for freq_ii, (fmin, fmax) in enumerate(self._freq_ranges):
            # Infer window size based on the frequency being used
            w_size = self.n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # filter the data at the desired frequency
            epochs_bp = epochs.copy()
            epochs_bp._data = filter_data(epochs_bp.get_data(), self.sfreq,
                                          fmin, fmax, verbose='CRITICAL')

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(self._centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # extract data for this time window
                s = self._transform(epochs_bp, y, w_tmin, w_tmax,
                                    method='score', pos=(freq_ii, t))
                self._scores[freq_ii, t] = s

        return self._scores
