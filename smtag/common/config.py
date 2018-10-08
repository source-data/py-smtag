# -*- coding: utf-8 -*-
#T. Lemberger, 2018
import os

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
    _scans_dir_name    = "scans"
    _img_grid_size     = 5 # grid size used to encode the location of elements on images
    _nbits             = 32 # number of features use to encode characters 
    _marking_char      = u'\uE000' # Substitution special xml-compatible character used to mark anonymized entities.
    _min_padding       = 20 # the number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
    _min_size          = 140 # input needs to be of minimal size to survive successive convergent convolutions; ideally, should be calculated analytically
    _default_threshold = 0.5 # threshold applied by default when descritizing predicted value and when considering a predicted value a 'hit' in accuracy calculation


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
    @property
    def scans_dir(self):
        """
        Path to results of hyperparameter scans.
        """
        return os.path.join(self.working_directory, self._scans_dir_name)
    @property
    def img_grid_size(self):
        """
        Grid size used to encode the location of elements on images.
        """
        return self._img_grid_size
    @property
    def nbits(self):
        """
        Number of features used to encode a character.
        """
        return self._nbits
    @property
    def marking_char(self):
        """
        Substitution special xml-compatible character used to mark anonymized entities.
        """
        return self._marking_char
    @property
    def min_padding(self):
        """
        The number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
        """
        return self._min_padding
    @property
    def min_size(self):
        """
        Input needs to be of minimal size to survive successive convergent convolutions; ideally, should be calculated analytically
        """
        return self._min_size
    @property
    def default_threshold(self):
        """
        Threshold applied by default when descritizing predicted value and when considering a predicted value a 'hit' in accuracy calculation
        """
        return self._default_threshold

config = Config()
