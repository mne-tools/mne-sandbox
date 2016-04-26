# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less)
from nose.tools import assert_raises, assert_equal
from copy import deepcopy

from mne_sandbox.connectivity import mvar_connectivity
from mne_sandbox.connectivity.mvar import _fit_mvar_lsq, _fit_mvar_yw


def _make_data(var_coef, n_samples, n_epochs):
    var_order = var_coef.shape[0]
    n_signals = var_coef.shape[1]

    x = np.random.randn(n_signals, n_epochs * n_samples + 10 * var_order)
    for i in range(var_order, x.shape[1]):
        for k in range(var_order):
            x[:, [i]] += np.dot(var_coef[k], x[:, [i - k - 1]])

    x = x[:, -n_epochs * n_samples:]

    win = np.arange(0, n_samples)
    return [x[:, i + win] for i in range(0, n_epochs * n_samples, n_samples)]


def test_mvar_connectivity():
    """Test MVAR connectivity estimation"""
    # Use a case known to have no spurious correlations (it would bad if
    # nosetests could randomly fail):
    np.random.seed(0)

    n_sigs = 3
    n_epochs = 100
    n_samples = 500

    # test invalid fmin fmax settings
    assert_raises(ValueError, mvar_connectivity, [], 'S', 5, fmin=10, fmax=5)
    assert_raises(ValueError, mvar_connectivity, [], 'DTF', 1, fmin=(0, 11),
                  fmax=(5, 10))
    assert_raises(ValueError, mvar_connectivity, [], 'PDC', 99, fmin=(11,),
                  fmax=(12, 15))
    assert_raises(ValueError, mvar_connectivity, [], 'S', fitting_mode='')
    assert_raises(NotImplementedError, mvar_connectivity, [], 'H',
                  fitting_mode='yw')

    methods = ['S', 'COH', 'DTF', 'PDC', 'ffDTF', 'GPDC', 'GDTF', 'A']

    # generate data without connectivity
    var_coef = np.zeros((1, n_sigs, n_sigs))
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs, p, p_vals = mvar_connectivity(data, methods, order=1,
                                              fitting_mode='yw')
    con = dict((m, c) for m, c in zip(methods, con))
    assert_equal(p, 1)

    assert_array_almost_equal(con['S'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0].diagonal(), np.ones(n_sigs))
    assert_array_almost_equal(con['DTF'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['PDC'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['ffDTF'][:, :, 0] / np.sqrt(len(freqs[0])),
                              np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['GPDC'][:, :, 0], np.eye(n_sigs), decimal=2)
    assert_array_almost_equal(con['GDTF'][:, :, 0], np.eye(n_sigs), decimal=2)

    # generate data with strong directed connectivity
    f = 1e3
    var_coef = np.zeros((1, n_sigs, n_sigs))
    var_coef[:, 1, 0] = f
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs, p, p_vals = mvar_connectivity(data, methods, order=(2, 5))
    con = dict((m, c) for m, c in zip(methods, con))

    h = var_coef.squeeze() + np.eye(n_sigs)

    assert_array_almost_equal(con['S'][:, :, 0] / f**2, np.dot(h, h.T) / f**2,
                              decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0], np.dot(h, h.T) > 0,
                              decimal=2)
    assert_array_almost_equal(con['DTF'][:, :, 0],
                              h / np.sum(h, 1, keepdims=True), decimal=2)
    assert_array_almost_equal(con['ffDTF'][:, :, 0] / np.sqrt(len(freqs[0])),
                              h / np.sum(h, 1, keepdims=True), decimal=2)
    assert_array_almost_equal(con['GDTF'][:, :, 0],
                              h / np.sum(h, 1, keepdims=True), decimal=2)
    assert_array_almost_equal(con['PDC'][:, :, 0],
                              h / np.sum(h, 0, keepdims=True), decimal=2)
    assert_array_almost_equal(con['GPDC'][:, :, 0],
                              h / np.sum(h, 0, keepdims=True), decimal=2)

    # generate data with strong cascaded directed connectivity
    f = 1e3
    var_coef = np.zeros((1, n_sigs, n_sigs))
    var_coef[:, 1, 0] = f
    var_coef[:, 2, 1] = f
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs, p, p_vals = mvar_connectivity(data, methods, order=(1, None))
    con = dict((m, c) for m, c in zip(methods, con))

    assert_array_almost_equal(con['S'][:, :, 0] / f**4, [[f**-4, f**-3, f**-2],
                                                         [f**-3, f**-2, f**-1],
                                                         [f**-2, f**-1, f**0]],
                              decimal=2)
    assert_array_almost_equal(con['COH'][:, :, 0], np.ones((n_sigs, n_sigs)),
                              decimal=2)
    assert_array_almost_equal(con['DTF'][:, :, 0], [[1, 0, 0],
                                                    [1, 0, 0],
                                                    [1, 0, 0]], decimal=2)
    assert_array_almost_equal(con['ffDTF'][:, :, 0] / np.sqrt(len(freqs[0])),
                              [[1, 0, 0], [1, 0, 0], [1, 0, 0]], decimal=2)
    assert_array_almost_equal(con['GDTF'], con['DTF'], decimal=2)

    h = var_coef.squeeze() + np.eye(n_sigs)
    assert_array_almost_equal(con['PDC'][:, :, 0],
                              h / np.sum(h, 0, keepdims=True), decimal=2)
    assert_array_almost_equal(con['GPDC'], con['PDC'], decimal=2)

    # generate data with some directed connectivity
    # check if statistics report only significant connectivity where the
    # original coefficients were non-zero
    var_coef = np.zeros((1, n_sigs, n_sigs))
    var_coef[:, 1, 0] = 1
    var_coef[:, 2, 1] = 1
    data = _make_data(var_coef, n_samples, n_epochs)

    con, freqs, p, p_vals = mvar_connectivity(data, 'PDC', order=(1, None),
                                              n_surrogates=20)

    for i in range(n_sigs):
        for j in range(n_sigs):
            if var_coef[0, i, j] > 0:
                assert_array_less(p_vals[0][i, j, 0], 0.05)
            else:
                assert_array_less(0.05, p_vals[0][i, j, 0])


def test_fit_mvar():
    """Test MVAR model fitting"""
    np.random.seed(0)

    n_sigs = 3
    n_epochs = 50
    n_samples = 200

    var_coef = np.zeros((1, n_sigs, n_sigs))
    var_coef[0, :, :] = [[0.9, 0, 0],
                         [1, 0.5, 0],
                         [2, 0, -0.5]]
    data = _make_data(var_coef, n_samples, n_epochs)
    data0 = deepcopy(data)

    var = _fit_mvar_lsq(data, pmin=1, pmax=1, delta=0, n_jobs=1, verbose=0)
    assert_array_equal(data, data0)
    assert_array_almost_equal(var_coef[0], var.coef, decimal=2)

    var = _fit_mvar_yw(data, pmin=1, pmax=1, n_jobs=1, verbose=0)
    assert_array_equal(data, data0)
    assert_array_almost_equal(var_coef[0], var.coef, decimal=2)
