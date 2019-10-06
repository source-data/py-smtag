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
    _dirignore         = ['.DS_Store'] # directories that should be ignored when scanning data or document compendia
    _allowed_img       = ['.jpg', '.jpeg', '.png']
    _img_grid_size     = 7 # grid size used to encode the location of elements on images
    _resized_img_size  = 512 # size of the resized image used for visual context
    _viz_context_features = 2208*7*7 # number of features used as visual context features; output of densenet161.features
    _ocr_max_edit_dist = 0.1 # max edit distance per character length between ocr term and matching term in caption
    _ocr_min_overlap   = 2 # minimum length of overlap between ocr term and caption term
    _nbits             = 8 # number of features use to encode characters; 31 for full unicode, 17 for emoji and greek; 7 for ASCII; WARNING should be a multiple of attention heads when multihead attention used
    _embedding_out_channels = 128 # the number of channels used for learned deep embeddings
    _marking_char      = '_' # Substitution special xml-compatible character used to mark anonymized entities.
    _masking_proba     = 1.0 # probability with wich an element selected to be potentially masked is effectively masked
    _padding_char      = '`' # " " # character used to padd strings; would be smarter to use character different from space
    _min_padding       = 100 # the number of (usually space) characters added to each example as padding to mitigate 'border effects' in learning
    _min_size          = 380 # input needs to be of minimal size to survive successive convergent convolutions with unet2 with 3 super layers and no padding; ideally, should be calculated analytically
    _default_threshold = 0.5 # threshold applied by default when descritizing predicted value and when considering a predicted value a 'hit' in accuracy calculation
    _fusion_threshold = 0.1 # threshold to allow adjascent token with identical features to be fused

    ############################################################################
    # MODELS
    #
    # WITH VISUAL CONTEXT
    # _model_entity_viz = "5X_L1200_fig_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-05-01-16-07.zip"
    # _model_geneprod_role_viz = "5X_L1200_geneprod_anonym_not_reporter_fig_intervention_assayed_2019-05-20-14-52.zip"
    # _model_molecule_role_viz = "5X_L1200_molecule_anonym_fig_intervention_assayed_2019-05-03-15-17.zip"
    # # no diseasee model with viz context because no traininset for this
    # no reporter model with viz because viz does not help
    # no panel_stop model with viz because viz does not help

    # WITHOUT VISUAL CONTEXT
    _model_entity_no_viz = "5X_L1200_article_embeddings_128_small_molecule_geneprod_subcellular_cell_tissue_organism_assay_2019-08-23-17-46.zip" # 
    _model_geneprod_reporter_no_viz = "5X_L1200_article_embeddings_128_reporter_2019-08-28-00-08_epoch_23_.zip"
    _model_geneprod_role_no_viz = "5X_L1200_anonym_not_reporter_article_embeddings_128_intervention_assayed_2019-08-22-16-25.zip"
    _model_molecule_role_no_viz = "5X_L1200_molecule_anonym_article_embeddings_128_intervention_assayed_2019-08-28-23-33_epoch_51.zip"
    _model_disease_no_viz = "10X_L1200_disease_articke_embeddings_128-5X_L1200_article_embeddings_128_disease_2019-08-25-21-47.zip"
    _model_panel_stop_no_viz = "5X_L1200_emboj_2012_no_viz_panel_stop_2019-08-29-08-31.zip"
    _embeddings_model = "article_embeddings_128.zip" # article_embedding_pmc.zip" #shuffle3_embedding_pmc.zip" # embeddings_verbs_pmc_abstracts.zip" # "shuffle3_embeddings.py" # "verbs_embeddings.zip" #

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
    def embeddings_dir(self):
        embedd_dir = os.path.join(self.working_directory, self._embeddings_dir_name)
        if not os.path.exists(embedd_dir):
            os.mkdir(embedd_dir)
        return embedd_dir
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
    def allowed_img(self):
        return self._allowed_img
    @property
    def img_grid_size(self):
        """
        Grid size used to encode the location of elements on images.
        """
        return self._img_grid_size
    @property
    def resized_img_size(self):
        """
        Size of the resized image used for visual context.
        """
        return self._resized_img_size
    @property
    def viz_cxt_features(self):
        """
        The number of visual context features used
        """
        return self._viz_context_features
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
    def embedding_out_channels(self):
        """
        Number of channels used for learned deep embeddings.
        """
        return self._embedding_out_channels
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
    # @property
    # def model_entity_viz(self):
    #     return self._model_entity_viz
    @property
    def model_entity_no_viz(self):
        return self._model_entity_no_viz
    # @property
    # def model_geneprod_role_viz(self):
    #     return self._model_geneprod_role_viz
    @property
    def model_geneprod_role_no_viz(self):
        return self._model_geneprod_role_no_viz
    @property
    def model_geneprod_reporter_no_viz(self):
        return self._model_geneprod_reporter_no_viz
    # @property
    # def model_molecule_role_viz(self):
    #     return self._model_molecule_role_viz
    @property
    def model_molecule_role_no_viz(self):
        return self._model_molecule_role_no_viz
    @property
    def model_panel_stop_no_viz(self):
        return self._model_panel_stop_no_viz
    @property
    def model_disease_no_viz(self):
        return self._model_disease_no_viz

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
