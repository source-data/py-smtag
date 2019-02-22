# -*- coding: utf-8 -*-
#T. Lemberger, 2018
import os
from .working_directory import WorkingDirectoryNotSetError
from .working_directory import fetch_working_directory, validated_working_directory

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
    _data_dir_name     = "data" # compendia of xml documents or brat annotations
    _encoded_dir_name  = "encoded" # examples with textual and visual information encoded
    _data4th_dir_name  = "data4th" # files generated by sampling and conversion to tensors of encoded examples
    _model_dir_name    = "models" # models saved during and after training
    _prod_dir_name     = "rack" # production models used by the SmartTag engine
    _log_dir_name      = "log" # general logging dir
    _runs_log_dir_name = "runs" # dir for tensorboard logs
    _scans_dir_name    = "scans" # results of hyperparameter scans

    ############################################################################
    # VARIABLES
    #
    _img_grid_size     = 3 # grid size used to encode the location of elements on images
    _k_pca_components = 10 # number of PCA components to reduce visual context features
    _fraction_images_pca_model = 0.1 # fraction of the visual context files to use to train the PCA model
    _nbits             = 32 # number of features use to encode characters; 31 for full unicode, 17 for emoji and greek; 7 for ASCII
    _marking_char      = u'\uE000' # Substitution special xml-compatible character used to mark anonymized entities.
    _padding_char      = " " # character used to padd strings; would be smarter to use character different from space
    _min_padding       = 20 # 380 # the number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
    _min_size          = 380 # input needs to be of minimal size to survive successive convergent convolutions with unet2 with 3 super layers and no padding; ideally, should be calculated analytically
    _default_threshold = 0.5 # threshold applied by default when descritizing predicted value and when considering a predicted value a 'hit' in accuracy calculation
    _fusion_threshold = 0.1 # threshold to allow adjascent token with identical features to be fused

    ############################################################################
    # MODELS
    #
    _model_assay = "10X_L400_all_large_padding_no_ocr_assay_2019-02-12-15-18.zip"
    _model_entity = "10X_L400_all_large_padding_no_ocr_small_molecule_geneprod_subcellular_cell_tissue_organism_2019-02-11-18-08.zip"
    _model_geneprod_role = "10X_L400_geneprod_anonym_not_reporter_large_padding_no_ocr_intervention_assayed_2019-02-11-23-22.zip"
    _model_geneprod_reporter = "10X_L400_geneprod_exclusive_padding_no_ocr_reporter_2019-02-12-10-57.zip"
    _model_molecule_role = "10X_L400_small_molecule_anonym_large_padding_no_ocr_intervention_assayed_2019-02-18-15-32.zip"
    _model_panel_stop = "10X_L1200_all_large_padding_no_ocr_panel_stop_2019-02-18-17-00.zip"
    _model_disease = "10X_L1200_NCBI_disease_augmented_large_padding_disease_2019-02-12-17-46.zip"

    def __init__(self):
        self.working_directory = fetch_working_directory()

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
        return os.path.join(self.working_directory, self._image_dir_name)
    @property
    def data_dir(self):
        return os.path.join(self.working_directory, self._data_dir_name)
    @property
    def data4th_dir(self):
        return os.path.join(self.working_directory, self._data4th_dir_name)
    @property
    def encoded_dir(self):
        return os.path.join(self.working_directory, self._encoded_dir_name)
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
    def weight(self):
        return self._weight
    @property
    def img_grid_size(self):
        """
        Grid size used to encode the location of elements on images.
        """
        return self._img_grid_size
    @property
    def k_pca_components(self):
        """
        The number of components of the PCA model used to reduce visual context features.
        """
        return self._k_pca_components
    @property
    def viz_cxt_features(self):
        """
        The number of visual context features used (the number of PCA components * positions on the image grid)
        """
        return self.k_pca_components * (self.img_grid_size ** 2)
    @property
    def fraction_images_pca_model(self):
        """
        Fraction of the available visual context files to use to train the PCA model that reduces visual context features.
        """
        return self._fraction_images_pca_model
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
    def model_assay(self):
        return self._model_assay
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
