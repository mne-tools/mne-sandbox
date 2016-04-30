# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: Simplified BSD

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_equal

from mne_sandbox.viz import (plot_connectivity_circle,
                             plot_connectivity_matrix,
                             plot_connectivity_inoutcircles)
from mne_sandbox.viz.connectivity import _plot_connectivity_matrix_nodename

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server


def test_plot_connectivity_circle():
    """Test plotting connecitvity circle
    """
    label_names = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
                   'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh']

    con = np.random.RandomState(42).rand(5, 5)

    plot_connectivity_circle(con, label_names, plot_names=True)
    plot_connectivity_circle(con, label_names, plot_names=False)


def test_plot_connectivity_matrix():
    """Test plotting connecitvity matrix
    """
    label_names = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
                   'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh']

    con = np.random.RandomState(42).rand(5, 5)
    con0 = con.copy()

    assert_raises(ValueError, plot_connectivity_matrix,
                  con=np.empty((2, 2, 2)), node_names=label_names)
    assert_raises(ValueError, plot_connectivity_matrix,
                  con=np.empty((1, 2)), node_names=label_names)

    plot_connectivity_matrix(con, label_names, colormap='jet', title='Test')

    # check that function does not change arguments
    assert_array_equal(con, con0)

    # test status bar text
    labels = ['a', 'b', 'c', 'd']
    con = np.empty((8, 8))
    con[:] = np.nan
    con[2:-2, 2:-2] = np.arange(16).reshape(4, 4)

    str1 = _plot_connectivity_matrix_nodename(1, 1, con, labels)
    str2 = _plot_connectivity_matrix_nodename(2, 3, con, labels)

    assert_equal(str1, '')
    assert_equal(str2, 'a --> b: 4.00')


def test_plot_connectivity_inoutcircles():
    """Test plotting directional connecitvity circles
    """
    label_names = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
                   'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh']

    con = np.random.RandomState(42).rand(5, 5)
    con0 = con.copy()

    assert_raises(ValueError, plot_connectivity_inoutcircles, con, 'n/a',
                  label_names)
    assert_raises(ValueError, plot_connectivity_inoutcircles, con, 99,
                  label_names)

    plot_connectivity_inoutcircles(con, 'bankssts-rh', label_names,
                                   title='Test')

    # check that function does not change arguments
    assert_array_equal(con, con0)
