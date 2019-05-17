import mne
import numpy as np
import scipy.stats
from numpy.testing import assert_allclose, assert_equal

from mne_sandbox.preprocessing.eog import eog_regression


def test_eog_regression():
    """Test EOG artifact removal using RAAA"""
    # Clean EEG signal: weak 10 Hz sine
    sine = 0.01 * np.sin(10 * 2 * np.pi * np.arange(410) / 100.)
    clean = mne.io.RawArray(
        data=np.vstack([
            np.tile(sine[np.newaxis, :], (2, 1)),
            np.zeros((3, 410)),  # EOG
        ]),
        info=mne.create_info(
            ['EEG1', 'EEG2', 'HEOG', 'VEOG', 'REOG'],
            sfreq=100.,
            ch_types=['eeg', 'eeg', 'eog', 'eog', 'eog'],
        ),
    )
    HEOG_ind = clean.ch_names.index('HEOG')
    VEOG_ind = clean.ch_names.index('VEOG')
    REOG_ind = clean.ch_names.index('REOG')
    events = np.array([
        [000, 0, 1],  # Blink
        [100, 0, 1],  # Blink
        [200, 0, 2],  # Horizontal saccade
        [300, 0, 3],  # Vertical saccade
    ])

    # Scenario 1: Some blinks captured by the VEOG channel
    blinks = clean.copy()
    blink_shape = scipy.stats.norm(0.5, 0.05).pdf(np.arange(0, 1, 0.01))
    blink_shape /= blink_shape.max()
    blink_shape = np.tile(blink_shape, 2)
    blinks._data[[0, 1, VEOG_ind], :200] += np.dot(
        [[1.1], [2.1], [1.0]],
        blink_shape[np.newaxis, :]
    )
    cleaned, weights = eog_regression(
        raw=blinks,
        blink_epochs=mne.Epochs(
            blinks, events, {'blink': 1},
            tmin=0, tmax=1, preload=True),
        eog_channels='VEOG',
        copy=True,
    )
    assert_allclose(cleaned._data[:2], clean._data[:2], atol=1e-12)
    assert_allclose(weights, [[1.1, 2.1]])

    # Scenario 2: Some blinks captured by VEOG and horizontal saccades captured
    #             by HEOG
    blink_sacc = clean.copy()
    # Add blinks
    blink_sacc._data[[0, 1, VEOG_ind], :200] += np.dot(
        [[1.1], [2.1], [1.0]],
        blink_shape[np.newaxis, :]
    )
    # Add saccades
    sacc_shape = np.hstack((
        scipy.stats.norm(0.2, 0.05).pdf(np.arange(0, 0.2, 0.01)),
        scipy.stats.norm(0.2, 0.20).pdf(np.arange(0.2, 1, 0.01))
    ))
    sacc_shape[:20] /= sacc_shape[:20].max()
    sacc_shape[20:] /= sacc_shape[20:].max()
    sacc_shape = np.tile(sacc_shape, 2)
    blink_sacc._data[[0, 1, HEOG_ind], 200:400] += np.dot(
        [[1.2], [2.2], [1.0]],
        sacc_shape[np.newaxis, :]
    )
    cleaned, weights = eog_regression(
        raw=blink_sacc,
        blink_epochs=mne.Epochs(
            blink_sacc, events, {'blink': 1},
            tmin=0, tmax=1, preload=True),
        saccade_epochs=mne.Epochs(
            blink_sacc, events, {'horiz-sacc': 2},
            tmin=0, tmax=1, preload=True),
        eog_channels=['VEOG', 'HEOG'],
        copy=True,
    )
    assert_allclose(cleaned._data[:2], clean._data[:2], atol=1e-3)
    assert_allclose(weights, [[1.1, 2.1], [1.2, 2.2]], atol=1e-3)

    # Scenario 3: Some blinks captured by REOG, some saccades captured
    #             by HEOG. EOG also cross-contaminated.
    full_eog = clean.copy()
    # Add saccades (also contaminate VEOG and REOG)
    full_eog._data[[0, 1, VEOG_ind, HEOG_ind, REOG_ind], 200:400] += np.dot(
        [[1.2], [2.2], [0], [1.0], [0]],
        sacc_shape[np.newaxis, :]
    )
    # Add blinks (also contaminate HEOG)
    full_eog._data[[0, 1, VEOG_ind, HEOG_ind, REOG_ind], :200] += np.dot(
        [[1.1], [2.1], [1.0], [0], [1.0]],
        blink_shape[np.newaxis, :]
    )
    cleaned, weights = eog_regression(
        raw=full_eog,
        blink_epochs=mne.Epochs(
            full_eog, events, {'blink': 1},
            tmin=0, tmax=1, preload=True),
        saccade_epochs=mne.Epochs(
            full_eog, events, {'horiz-sacc': 2, 'vert-sacc': 3},
            tmin=0, tmax=1, preload=True),
        reog='REOG',
        eog_channels=['VEOG', 'HEOG', 'REOG'],
        copy=True,
    )
    assert_allclose(cleaned._data[:2], clean._data[:2], atol=1e-3)
    assert_allclose(weights, [[0, 0], [1.2, 2.2], [1.1, 2.1]], atol=1e-3)

    # We use the last scenario to run a few more tests
    raw = full_eog
    blink_epochs = mne.Epochs(
        full_eog, events, {'blink': 1},
        tmin=0, tmax=1, preload=True)
    saccade_epochs = mne.Epochs(
        full_eog, events, {'horiz-sacc': 2, 'vert-sacc': 3},
        tmin=0, tmax=1, preload=True)

    # Default parameters
    raw2 = raw.copy()
    raw3, weights = eog_regression(raw2, blink_epochs)
    assert_equal(raw2, raw3)
    assert_equal(weights.shape, (3, 2))

    # Picks parameter
    _, weights = eog_regression(raw, blink_epochs, saccade_epochs, copy=True,
                                picks=[0])
    assert_equal(weights.shape, (3, 1))

    # REOG parameter
    _, weights = eog_regression(raw, blink_epochs, saccade_epochs, copy=True,
                                reog='REOG', eog_channels=['VEOG', 'HEOG'])
    assert_equal(weights.shape, (3, 2))

    # EOG channels parameter
    _, weights = eog_regression(raw, blink_epochs, saccade_epochs, copy=True,
                                eog_channels='VEOG')
    assert_equal(weights.shape, (1, 2))

    # Order of the EOG channels should not matter
    cleaned, weights = eog_regression(raw, blink_epochs, saccade_epochs,
                                      copy=True, reog='REOG',
                                      eog_channels=['VEOG', 'REOG', 'HEOG'])
    assert_allclose(cleaned._data[:2], clean._data[:2], atol=1e-3)
    assert_allclose(weights, [[0, 0], [1.1, 2.1], [1.2, 2.2]], atol=1e-3)
