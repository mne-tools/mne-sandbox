# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

DEFAULTS = dict(
    bad_channels_faster=dict(
        use_metrics=['variance', 'correlation', 'hurst', 'kurtosis',
                     'line_noise'],
        thresh=3,
        max_iter=1,
        eeg_ref=None,
    ),
    bad_epochs_faster=dict(
        use_metrics=['amplitude', 'variance', 'deviation'],
        thresh=3,
        max_iter=1,
    ),
    bad_channels_in_epochs_faster=dict(
        use_metrics=['amplitude', 'variance', 'deviation', 'median_gradient',
                     'line_noise'],
        thresh=3,
        max_iter=1,
    ),
)


def _handle_default(k, v=None):
    """Helper to avoid dicts as default keyword arguments

    Use this function instead to resolve default dict values. Example usage::

        scalings = _handle_default('scalings', scalings)

    """
    this_mapping = deepcopy(DEFAULTS[k])
    if v is not None:
        if isinstance(v, dict):
            this_mapping.update(v)
        else:
            for key in this_mapping.keys():
                this_mapping[key] = v
    return this_mapping
