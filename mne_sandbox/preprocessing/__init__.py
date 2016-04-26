from .eog import eog_regression
from ._dss import dss
from ._sns import SensorNoiseSuppression
from .bads import (find_outliers,
                   find_bad_channels, find_bad_epochs,
                   find_bad_channels_in_epochs)
