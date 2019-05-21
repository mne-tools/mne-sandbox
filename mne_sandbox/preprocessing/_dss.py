# -*- coding: utf-8 -*-
"""Denoising source separation."""

# Authors: Daniel McCloy <drmccloy@uw.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from mne import (Epochs, EpochsArray, compute_covariance, create_info,
                 compute_rank)
from mne.cov import compute_whitener
from mne.decoding.receptive_field import _delay_time_series
from mne.utils import _ensure_int


def dss(data, data_max_components=None, data_thresh='auto',
        bias_max_components=None, bias_thresh=1e-6, max_delay=0,
        rank=None, return_data=True):
    """Process physiological data with denoising source separation (DSS).

    Implementation follows the procedure described in Särelä & Valpola [1]_
    and de Cheveigné & Simon [2]_.

    Parameters
    ----------
    data : instance of Epochs | array of shape (n_trials, n_channels, n_times)
        Data to be denoised.
    data_max_components : int | None
        Maximum number of components to keep during PCA decomposition of the
        data. ``None`` (the default) keeps all suprathreshold components.
    data_thresh : float | str | None
        Threshold to pass to :func:`mne.compute_rank`.
    bias_max_components : int | None
        Maximum number of components to keep during PCA decomposition of the
        bias function. ``None`` (the default) keeps all suprathreshold
        components.
    bias_thresh : float | None
        Threshold (relative to the largest component) below which components
        will be discarded during decomposition of the bias function. ``None``
        (the default) keeps all non-zero values; to keep all values, pass
        ``thresh=None`` and ``max_components=None``.
    max_delay : int
        Maximum delay (in samples) to consider. Zero will use DSS, anything
        greater than zero will use tsDSS.
    rank : None | dict | 'info' | 'full'
        See :func:`mne.compute_rank`.
    return_data : bool
        Whether to return the denoised data along with the denoising matrix,
        which can be obtained via::

            data_dss = np.einsum('ij,hjk->hik', dss_mat, data)

    Returns
    -------
    dss_mat : array of shape (n_dss_components, n_channels)
        The denoising matrix. Apply to data via ``np.dot(dss_mat, ep)``, where
        ``ep`` is an epoch of shape (n_channels, n_samples).
    dss_data : array of shape (n_trials, n_dss_components, n_samples)
        The denoised data. Note that the DSS components are orthogonal virtual
        channels and may be fewer in number than the number of channels in the
        input Epochs object. Returned only if ``return_data`` is ``True``.

    References
    ----------
    .. [1] Särelä, Jaakko, and Valpola, Harri (2005). Denoising source
       separation. Journal of Machine Learning Research 6: 233–72.
    .. [2] de Cheveigné, Alain, and Simon, Jonathan Z. (2008). Denoising based
       on spatial filtering. Journal of Neuroscience Methods, 171(2): 331-339.
    """
    if isinstance(data, (Epochs, EpochsArray)):
        epochs = data
        data = epochs.get_data()
    elif isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError('Data to denoise must have shape '
                             '(n_trials, n_channels, n_times).')
        info = create_info(data.shape[1], 1000., 'eeg')
        epochs = EpochsArray(data, info)
    else:
        raise TypeError('Data to denoise must be an instance of mne.Epochs or '
                        'ndarray, got type %s' % (type(data),))
    _check_thresh(data_thresh)
    rank = compute_rank(epochs, rank=rank, tol=data_thresh)

    # Upsample to virtual channels for tsDSS
    max_delay = _ensure_int(max_delay)
    if max_delay < 0:
        raise ValueError('max_delay must be ≥ 0, got %s' % (max_delay,))
    rank = {key: max_delay * val for key, val in rank.items()}
    data = _delay_time_series(data.transpose(2, 0, 1), 0, max_delay, 1.)
    data = data.transpose(1, 2, 3, 0)  # ep, ch, del, time
    data = np.reshape(data, (data.shape[0], -1, data.shape[-1]))
    info = epochs.info.copy()
    chs = list()
    for ch in info['chs']:
        for ii in range(max_delay + 1):
            this_ch = deepcopy(ch)
            this_ch['ch_name'] += '_%06d' % (ii,)
            chs.append(this_ch)
    info.update(projs=[], chs=chs)
    info._update_redundant()
    info._check_consistency()
    epochs = EpochsArray(data, info)

    # Actually compute DSS transformation
    dss_mat = _dss(epochs, rank, data_max_components,
                   bias_max_components, bias_thresh)
    if return_data:
        # next line equiv. to: np.array([np.dot(dss_mat, ep) for ep in data])
        dss_data = np.einsum('ij,hjk->hik', dss_mat, data)
        return dss_mat, dss_data
    else:
        return dss_mat


def _dss(epochs, rank, data_max_components,
         bias_max_components, bias_thresh):
    """Process physiological data with denoising source separation (DSS).

    Acts on covariance matrices; allows specification of arbitrary bias
    functions (as compared to the public ``dss`` function, which forces the
    bias to be the evoked response).
    """
    data_cov = compute_covariance(epochs, verbose='error')
    bias_epochs = EpochsArray(epochs.average(picks='all').data[np.newaxis],
                              epochs.info)
    bias_cov = compute_covariance(bias_epochs, verbose='error')

    # From Eq. 7 in the de Cheveigné and Simon paper, the dss_mat A:
    #     A = P @ Q @ R2 @ N2 @ R1 @ N1
    # Where:
    # - N1 is the initial normalization (row normalization)
    # - R1 the first PCA rotation
    # - N2 the second normalization (whitening)
    # - R2 the second PCA rotation
    # - Q the criterion-based selector
    # - P the projection back to sensor space

    # Here we skip N1, assume compute_whitener does a good enough job
    # of whitening the data that a diagonal prewhitening is not necessary.

    # First rotation (R1) and second normalization (N2)
    # -------------------------------------------------
    # Obtain via compute_whitener in MNE so we can be careful about channel
    # types and rank deficiency.
    N2_R1 = compute_whitener(data_cov, epochs.info, pca=True, rank=rank,
                             verbose='error')[0]  # ignore avg ref warnings

    # Second rotation (R2)
    # --------------------
    # bias covariance projected into whitened PCA space of data channels
    bias_cov_white = N2_R1 @ bias_cov['data'] @ N2_R1.T
    # proj. matrix from whitened data space to a space maximizing bias fxn
    _, R2 = _pca(bias_cov_white, bias_max_components, bias_thresh)
    R2 = R2.T
    # proj. matrix from data to bias-maximizing space (DSS space)
    A = R2 @ N2_R1
    # Normalize DSS dimensions
    P = np.sqrt(1 / np.diag(A.dot(data_cov['data']).dot(A.T)))
    A *= P[:, np.newaxis]

    # Q and P are left to be computed by the user.
    return A


def _check_thresh(thresh):
    if thresh is not None and thresh != 'auto' and not 0 <= thresh <= 1:
        raise ValueError('Threshold must be between 0 and 1 (or None).')


def _pca(cov, max_components=None, thresh=0):
    """Perform PCA decomposition.

    Parameters
    ----------
    cov : array-like
        Covariance matrix
    max_components : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).
    thresh : float | None
        Threshold (relative to the largest component) above which components
        will be kept. The default keeps all non-zero values; to keep all
        values, specify ``thresh=None`` and ``max_components=None``.

    Returns
    -------
    eigval : array
        1-dimensional array of eigenvalues.
    eigvec : array
        2-dimensional array of eigenvectors.
    """
    _check_thresh(thresh)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.abs(eigval)
    sort_ix = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, sort_ix]
    eigval = eigval[sort_ix]
    if max_components is not None:
        eigval = eigval[:max_components]
        eigvec = eigvec[:, :max_components]
    if thresh is not None:
        suprathresh = np.where(eigval / eigval.max() > thresh)[0]
        eigval = eigval[suprathresh]
        eigvec = eigvec[:, suprathresh]
    return eigval, eigvec
