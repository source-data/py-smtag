# -*- coding: utf-8 -*-
#T. Lemberger, 2018
import os
import argparse
from .working_directory import (
    WorkingDirectoryNotSetError,
    fetch_working_directory,
    validated_working_directory,
    WORKING_DIRECTORY_CLI_FLAG_NAME,
    WORKING_DIRECTORY_CLI_FLAG_SHORTNAME,
)
from .production_directory import (
    ProductionDirectoryNotSetError,
    fetch_production_directory,
    validated_production_directory,
    PRODUCTION_DIRECTORY_CLI_FLAG_NAME,
    PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME,
)

class Config():
    """
    Class that collects all configuration options.
    It allows to dynamically adjust its parameters at runtime.
    It should be treated as a singleton, in the sense that there will only be
    one instance of the class that is ever used. It is instantiated in the
    __init__.py of this package (folder), but will be aliased at the root level
    of the the whole library (ie in smtag/__init__.py)
    """
    ############################################################################
    # DIRECTORIES
    #
    _image_dir_name      = "img" # stock of unique images linked to documents
    _data_dir_name       = "data" # compendia of xml documents or brat annotations
    _data4th_dir_name    = "data4th" # files generated by sampling and conversion to tensors of encoded examples
    _model_dir_name      = "models" # models saved during and after training
    _prod_dir_name       = "rack" # production models used by the SmartTag engine
    _log_dir_name        = "log" # general logging dir
    _runs_log_dir_name   = "runs" # dir for tensorboard logs
    _scans_dir_name      = "scans" # results of hyperparameter scans
    _embeddings_dir_name = "embeddings" # pretrained networks generating context-aware character-level embeddings

    ############################################################################
    # VARIABLES
    #
    _cache_dataset     = 1024 # size of the cache used in Dataset to cache individual examples that will be packaged into a minibatch
    _dirignore         = ['.DS_Store', '__MACOSX'] # directories that should be ignored when scanning data or document compendia
    nbits             = int(os.getenv('NBITS')) # number of features use to encode characters; 31 for full unicode, 17 for emoji and greek; 7 for ASCII; WARNING should be a multiple of attention heads when multihead attention used
    _marking_char      = '_' # Substitution special xml-compatible character used to mark anonymized entities.
    _masking_proba     = 1.0 # probability with wich an element selected to be potentially masked is effectively masked
    _padding_char      = '`' # " " # character used to padd strings; would be smarter to use character different from space
    _min_padding       = 100 # the number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
    _min_size          = 380 # input needs to be of minimal size to survive successive convergent convolutions with unet2 with 3 super layers and no padding; ideally, should be calculated analytically
    _default_threshold = 0.5 # threshold applied by default when descritizing predicted value and when considering a predicted value a 'hit' in accuracy calculation
    _fusion_threshold = 0.1 # threshold to allow adjascent token with identical features to be fused

    ############################################################################
    # MODELS
    _model_entity = "2020-02-29-13-10_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_epoch_019.zip"
    _model_geneprod_role = "2020-02-29-22-47_intervention_assayed_epoch_019.zip"
    _model_molecule_role = "2020-03-01-01-29_intervention_assayed_epoch_019.zip"
    _model_geneprod_reporter = "2020-03-01-08-07_reporter_epoch_004.zip"
    _model_disease = "2020-03-01-08-49_disease_epoch_020.zip"
    _model_panel_stop = "2020-03-01-09-21_panel_stop_epoch_012.zip"
    _embeddings_model = "2020-02-24-01-31_last_saved.zip"

    def __init__(self):
        self.working_directory = fetch_working_directory()
        self.prod_dir = fetch_production_directory()

    @property
    def prod_dir(self):
        if self.__production_directory is None:
            raise ProductionDirectoryNotSetError
        return self.__production_directory

    @prod_dir.setter
    def prod_dir(self, new_production_directory):
        self.__production_directory = validated_production_directory(new_production_directory)

    @property
    def working_directory(self):
        if self.__working_directory is None:
            raise WorkingDirectoryNotSetError
        return self.__working_directory

    @working_directory.setter
    def working_directory(self, new_working_directory):
        self.__working_directory = validated_working_directory(new_working_directory)

    @property
    def image_dir(self):
        image_dir = os.path.join(self.working_directory, self._image_dir_name)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        return image_dir

    @property
    def data_dir(self):
        data_dir = os.path.join(self.working_directory, self._data_dir_name)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        return data_dir

    @property
    def data4th_dir(self):
        data4th_dir = os.path.join(self.working_directory, self._data4th_dir_name)
        if not os.path.exists(data4th_dir):
            os.mkdir(data4th_dir)
        return data4th_dir

    @property
    def model_dir(self):
        model_dir = os.path.join(self.working_directory, self._model_dir_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        return model_dir

    @property
    def runs_log_dir(self):
        runs_log_dir = os.path.join(self.working_directory, self._runs_log_dir_name)
        if not os.path.exists(runs_log_dir):
            os.mkdir(runs_log_dir)
        return runs_log_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.working_directory, self._log_dir_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return log_dir
    @property

    def scans_dir(self):
        """
        Path to results of hyperparameter scans.
        """
        scans_dir = os.path.join(self.working_directory, self._scans_dir_name)
        if not os.path.exists(scans_dir):
            os.mkdir(scans_dir)
        return scans_dir

    @property
    def embeddings_model(self):
        return self._embeddings_model

    @property
    def cache_dataset(self):
        """
        Size of the cache used in Dataset
        """
        return self._cache_dataset

    @property
    def dirignore(self):
        """
        List of directory names that should be ignored when scanning for datasets
        """
        return self._dirignore

    @property
    def marking_char(self):
        """
        Substitution special xml-compatible character used to mark anonymized entities.
        """
        return self._marking_char

    @property
    def masking_proba(self):
        """
        Probability with wich an element selected for potential masking will actually be masked.
        """
        return self._masking_proba

    @property
    def padding_char(self):
        """
        Special character used to pad strings.
        """
        return self._padding_char

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

    @property
    def fusion_threshold(self):
        """
        Threshold to allow adjascent token with identical features to be fused
        """
        return self._fusion_threshold

    @property
    def model_entity(self):
        return self._model_entity

    @property
    def model_geneprod_role(self):
        return self._model_geneprod_role

    @property
    def model_geneprod_reporter(self):
        return self._model_geneprod_reporter

    @property
    def model_molecule_role(self):
        return self._model_molecule_role

    @property
    def model_panel_stop(self):
        return self._model_panel_stop

    @property
    def model_disease(self):
        return self._model_disease

    def create_argument_parser_with_defaults(self, description=None):
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
        ################################################################################################################
        # --working_direcotory -w
        # .common.config.working_directory is dealing with this argument, and it uses sys.argv to do that, however it is
        # important that we define it as an available parameter here, otherwise argparse will complain if it ever gets used
        #
        parser.add_argument(WORKING_DIRECTORY_CLI_FLAG_SHORTNAME, WORKING_DIRECTORY_CLI_FLAG_NAME, help='Specify the working directory where to find special directories such as prod, data4th etc')

        # TODO: move to smtag.precit.engine
        # this parser argument should not be global, it is specific of the prediction packade and it should be moved there
        # this would require removing `docopt` and converting it to the standard `argparse`
        parser.add_argument(PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME, PRODUCTION_DIRECTORY_CLI_FLAG_NAME, help='Specify the production directory (a.k.a. the `rack` folder) where the trained models to be used for inference are stored')

        return parser
