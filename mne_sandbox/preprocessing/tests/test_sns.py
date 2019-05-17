# -*- coding: utf-8 -*-

import os.path as op

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from mne import create_info, io, pick_types, pick_channels
from mne.io import RawArray
from mne.utils import run_tests_if_main
from mne.datasets import testing

from mne_sandbox.preprocessing import SensorNoiseSuppression

data_path = testing.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_sns():
    """Test sensor noise suppression"""
    # artificial (IID) data
    data = np.random.RandomState(0).randn(102, 5000)
    info = create_info(len(data), 1000., 'mag')
    raw = io.RawArray(data, info)
    pytest.raises(ValueError, SensorNoiseSuppression, 'foo')
    pytest.raises(TypeError, SensorNoiseSuppression(10).fit, 'foo')
    pytest.raises(ValueError, SensorNoiseSuppression, -1)
    raw.info['bads'] = [raw.ch_names[1]]
    pytest.raises(ValueError, SensorNoiseSuppression(101).fit, raw)
    for n_neighbors, bounds in ((2, (17, 20)),
                                (5, (11, 15),),
                                (10, (9, 12)),
                                (20, (7, 10)),
                                (50, (5, 9)),
                                (100, (5, 8)),
                                ):
        sns = SensorNoiseSuppression(n_neighbors)
        sns.fit(raw)
        raw_sns = sns.apply(raw.copy())
        operator = sns.operator
        # bad channels are modified but not used
        assert_allclose(operator[:, 1], 0.)
        assert (np.sum(np.abs(operator)) > 0)
        assert_equal(operator[0].astype(bool).sum(), n_neighbors)
        assert_allclose(np.diag(operator), 0.)
        picks = pick_types(raw.info)
        orig_power = np.linalg.norm(raw[picks][0])
        # Test the suppression factor
        factor = orig_power / np.linalg.norm(raw_sns[picks][0])
        assert bounds[0] < factor < bounds[1]
    # degenerate conditions
    pytest.raises(TypeError, sns.apply, 'foo')
    sub_raw = raw.copy().pick_channels(raw.ch_names[:-1])
    pytest.raises(RuntimeError, sns.apply, sub_raw)  # not all orig chs
    sub_sns = SensorNoiseSuppression(8)
    sub_sns.fit(sub_raw)
    pytest.raises(RuntimeError, sub_sns.apply, raw)  # not all new chs
    # sample data
    raw = io.read_raw_fif(raw_fname)
    n_neighbors = 8
    sns = SensorNoiseSuppression(n_neighbors=n_neighbors)
    sns.fit(raw)
    raw_sns = sns.apply(raw.copy().load_data())
    operator = sns.operator
    # bad channels are modified
    assert_equal(len(raw.info['bads']), 2)
    for pick in pick_channels(raw.ch_names, raw.info['bads']):
        expected = np.zeros(operator.shape[0])
        sub_pick = sns._used_chs.index(raw.ch_names[pick])
        expected[sub_pick] = 1.
        pytest.raises(AssertionError, assert_allclose, operator[sub_pick],
                      expected)
        pytest.raises(AssertionError, assert_allclose, raw[pick][0],
                      raw_sns[pick][0])
    assert_equal(operator[0].astype(bool).sum(), n_neighbors)
    assert_equal(operator[0, 0], 0.)
    picks = pick_types(raw.info)
    orig_power = np.linalg.norm(raw[picks][0])
    # Test the suppression factor
    factor = orig_power / np.linalg.norm(raw_sns[picks][0])
    bounds = (1.3, 1.7)
    assert bounds[0] < factor < bounds[1]
    # degenerate conditions
    pytest.raises(RuntimeError, sns.apply, raw)  # not preloaded

    # Test against NoiseTools
    rng = np.random.RandomState(0)
    n_channels = 9
    x = rng.randn(n_channels, 10000)
    # make some correlations
    x = np.dot(rng.randn(n_channels, n_channels), x)
    x -= x.mean(axis=-1, keepdims=True)
    # savemat('test.mat', dict(x=x))  # --> run through nt_sns in MATLAB
    nt_op = np.array([  # Obtained from NoiseTools 18-Nov-2016
        [0, 0, -0.3528, 0, 0.6152, 0, 0, -0.3299, 0.1914],
        [0, 0, 0, 0.0336, 0, 0, -0.4284, 0, 0],
        [-0.2928, 0.2463, 0, 0, -0.0891, 0, 0, 0.2200, 0],
        [0, 0.0191, 0, 0, -0.3756, -0.3253, 0.4047, -0.4608, 0],
        [0.3184, 0, -0.0878, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.5865, 0, -0.2137],
        [0, -0.5190, 0, 0.5059, 0, 0.8271, 0, 0, -0.0771],
        [-0.3953, 0, 0.3092, -0.5018, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.2050, 0, 0, 0]]).T

    raw = RawArray(x, create_info(n_channels, 1000.,
                                  ['grad', 'grad', 'mag'] * 3))
    sns = SensorNoiseSuppression(3)
    sns.fit(raw)
    # this isn't perfect, but it's close
    np.testing.assert_allclose(sns.operator, nt_op, rtol=5e-2, atol=1e-3)


run_tests_if_main()
