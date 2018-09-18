# -*- coding: utf-8 -*-
#T. Lemberger, 2018
import os


NBITS = 32 # number of bits to encode characters; not very useful since making converter flexible slows it down
MARKING_CHAR = u'\u0000' # special character used to anonymize an entity before learning context-dependent features
MARKING_CHAR_ORD = ord(MARKING_CHAR)
MIN_PADDING = 20 # the number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
MIN_SIZE = 140 # input needs to be of minimal size to survive successive convergent convolutions; ideally, should be calculated analytically
DEFAULT_THRESHOLD = 0.5 # threshold applied by default when descritizing predicted value and when considering a predicted value a 'hit' in accuracy calculation

class Config():
    """
    Class that collects all configuration options.
    It allows to dynamically adjust its parameters at runtime.
    It should be treated as a singleton, in the sense that there will only be
    one instance of the class that is ever used. It is instantiated here, after
    the class declaration, but will be aliased at the root level of the package
    (ie in smtag/__init__.py)
    """
    _data_dir_name     = "data"
    _data4th_dir_name  = "data4th"
    _model_dir_name    = "models"
    _prod_dir_name     = "rack"
    _log_dir_name      = "log"
    def __init__(self):
        self.working_directory = "."

    @property
    def data_dir(self):
        return os.path.join(self.working_directory, self._data_dir_name)
    @property
    def data4th_dir(self):
        return os.path.join(self.working_directory, self._data4th_dir_name)
    @property
    def model_dir(self):
        return os.path.join(self.working_directory, self._model_dir_name)
    @property
    def prod_dir(self):
        return os.path.join(self.working_directory, self._prod_dir_name)
    @property
    def runs_log_dir(self):
        return os.path.join(self.working_directory, self._runs_log_dir_name)
    @property
    def log_dir(self):
        return os.path.join(self.working_directory, self._log_dir_name)

config = Config()
