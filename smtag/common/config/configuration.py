# -*- coding: utf-8 -*-
#T. Lemberger, 2018
import os
import argparse
from .working_directory import WorkingDirectoryNotSetError
from .working_directory import fetch_working_directory, validated_working_directory
from .working_directory import WORKING_DIRECTORY_CLI_FLAG_NAME, WORKING_DIRECTORY_CLI_FLAG_SHORTNAME

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
    _dirignore         = ['.DS_Store'] # directories that should be ignored when scanning data or document compendia
    _allowed_img       = ['.jpg', '.jpeg', '.png']
    _img_grid_size     = 7 # grid size used to encode the location of elements on images
    _k_pca_components  = 3 # number of PCA components to reduce visual context features
    _fraction_images_pca_model = 0.1 # fraction of the visual context files to use to train the PCA model
    _ocr_max_edit_dist = 0.5 # max edit distance per character length between ocr term and matching term in caption
    _ocr_min_overlap   = 2 # minimum lenght of overlap between ocr term and caption term
    _nbits             = 17 # number of features use to encode characters; 31 for full unicode, 17 for emoji and greek; 7 for ASCII
    _marking_char      = u'\uE000' # Substitution special xml-compatible character used to mark anonymized entities.
    _padding_char      = " " # character used to padd strings; would be smarter to use character different from space
    _min_padding       = 380 # the number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
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
    def encoded_dir(self):
        encoded_dir = os.path.join(self.working_directory, self._encoded_dir_name)
        if not os.path.exists(encoded_dir):
            os.mkdir(encoded_dir)
        return encoded_dir
    @property
    def model_dir(self):
        model_dir = os.path.join(self.working_directory, self._model_dir_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        return model_dir
    @property
    def prod_dir(self):
        prod_dir = os.path.join(self.working_directory, self._prod_dir_name)
        if not os.path.exists(prod_dir):
            os.mkdir(prod_dir)
        return prod_dir
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
    def dirignore(self):
        """
        List of directory names that should be ignored when scanning for datasets
        """
        return self._dirignore
    @property
    def allowed_img(self):
        return self._allowed_img
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
    def ocr_max_edit_dist(self):
        """
        Max edit distance per character length between ocr term and matching term in caption
        """
        return self._ocr_max_edit_dist
    @property
    def ocr_min_overlap(self):
        """
        Minimum length of overlap between ocr term and caption term
        """
        return self._ocr_min_overlap
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

    def create_argument_parser_with_defaults(self, description=None):
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
        ################################################################################################################
        # --working_direcotory -w
        # .common.config.working_directory is dealing with this argument, and it uses sys.argv to do that, however it is
        # important that we define it as an available parameter here, otherwise argparse will complain if it ever gets used
        #
        parser.add_argument(WORKING_DIRECTORY_CLI_FLAG_SHORTNAME, WORKING_DIRECTORY_CLI_FLAG_NAME, help='Specify the working directory where to find special directories such as rack, prod, data4th etc')

        return parser
