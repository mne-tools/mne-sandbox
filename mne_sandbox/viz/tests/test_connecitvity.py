# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: Simplified BSD

import numpy as np
from numpy.testing import assert_array_equal

from mne_sandbox.viz import (plot_connectivity_circle,
                             plot_connectivity_matrix,
                             plot_connectivity_inoutcircles)

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

    plot_connectivity_matrix(con, label_names)

    # check that function does not change arguments
    assert_array_equal(con, con0)


def test_plot_connectivity_inoutcircles():
    """Test plotting directional connecitvity circles
    """
    label_names = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
                   'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh']

    con = np.random.RandomState(42).rand(5, 5)
    con0 = con.copy()

    plot_connectivity_inoutcircles(con, 'bankssts-rh', label_names)

    # check that function does not change arguments
    assert_array_equal(con, con0)
