"""
Config.prod_dir is now configurable a la working_directory

this allows to specify a rack folder that doesnt necessarily live inside of the packed smtag library.

this is convenient for the cases in which the trained moldes are too big (eg densenet) and cannot be commited to the git repository.

in this scenarios the user of py-smtag would need to manually download the models and point to them with the new env variable `SMTAG_PRODUCTION_DIRECTORY` or by using the new CLI flags "--production_directory and its shorted version -p
"""

import os
import sys
import logging
from .errors import ConfigError

PRODUCTION_DIRECTORY_ENV_VAR_NAME = "SMTAG_PRODUCTION_DIRECTORY"
PRODUCTION_DIRECTORY_CLI_FLAG_NAME = "--production_directory"
PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME = "-p"

DEFAULT_PRODUCTION_DIRECTORY_NAME = "rack"
def default_production_directory():
    """
    Returns the absolute path to the default production directory to use if the user doesn't specify otherwise
    The default production directory name is configure via the constant `DEFAULT_PRODUCTION_DIRECTORY_NAME`
    This directory can be found at the root of the project (e.g. where the setup.py file is)
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(dir_path, "..", "..", "..")
    prod_dir = os.path.join(root_path, DEFAULT_PRODUCTION_DIRECTORY_NAME)
    prod_dir = os.path.abspath(prod_dir)
    return prod_dir


PRODUCTION_DIRECTORY_HELP_MESSAGE = f"""
            In order to configure you production directory use one of this options, sorted by lesser precedence:
            1) Set the environment variable `{PRODUCTION_DIRECTORY_ENV_VAR_NAME}` (check your .env file)
               Examples:
                {PRODUCTION_DIRECTORY_ENV_VAR_NAME}='/absolute/path/to/resources_folder' python -m smtag.predict.egine --demo
                {PRODUCTION_DIRECTORY_ENV_VAR_NAME}='./relative/path/to/resources_folder' python -m smtag.predict.egine --demo
            2) Set the `{PRODUCTION_DIRECTORY_CLI_FLAG_NAME}` flag or its abbreviated version `{PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME}`
               Examples:
                python -m smtag.predict.egine --demo {PRODUCTION_DIRECTORY_CLI_FLAG_NAME} "/absolute/path/to/resources_folder"
                python -m smtag.predict.egine --demo {PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME} "./relative/path/to/resources_folder"
            3) Set the `smtag.config.prod_dir` programatically
               Examples:
                import smtag
                smtag.config.prod_dir = "/absolute/path/to/resources_folder"
                smtag.config.prod_dir = "./relative/path/to/resources_folder"

            If none of the above options is specified, the default directory is assumed to be:

                {default_production_directory()}
            """
class ProductionDirectoryNotSetError(ConfigError):
    def __init__(self):
        super(ProductionDirectoryNotSetError, self).__init__()
        logging.error(f"""
            #######################################################################################################
            #
            ERROR: PRODUCTION DIRECTORY NOT SET

            {PRODUCTION_DIRECTORY_HELP_MESSAGE}

            #
            #######################################################################################################
            """)
class ProductionDirectoryDoesNotExistError(ConfigError):
    def __init__(self, path):
        super(ProductionDirectoryDoesNotExistError, self).__init__()
        logging.error(f"""
            #######################################################################################################
            #
            ERROR: PRODUCTION DIRECTORY DOES NOT EXIST

            The specified production directory does not exist:

                {path}

            {PRODUCTION_DIRECTORY_HELP_MESSAGE}

            #
            #######################################################################################################
        """)

def fetch_production_directory():
    """
    Returns the working directory path if specified by the user.
    If specified in different ways the order of precedence is:

        1. the CLI flag --production_directory
        2. a global env variable with the name specified by `PRODUCTION_DIRECTORY_ENV_VAR_NAME`
        3. the default folder

    Returns None otherwise
    """
    return (
        __fetch_production_dir_from_flag() or
        __fetch_production_dir_from_env() or
        default_production_directory() or
        None
    )


def validated_production_directory(path):
    """
    Sanitzes the give path and makes sure it exists
    Receives:
        @param path: string specifying a path to the working directory

    Returns:
        The absolute path

    Raises:
        ProductionDirectoryDoesNotExistError if the path points to something that is not a directory
        or if it does not exist
    """
    if path is None:
        return None
    # convert to absolute path
    path =  os.path.abspath(path)
    # check that the directory exists and it is a directory
    if not os.path.isdir(path):
        raise ProductionDirectoryDoesNotExistError(path)
    return path


def __fetch_production_dir_from_env():
    """
    Returns:
        - The value set in the environment variable with the name specified by
        the constant PRODUCTION_DIRECTORY_ENV_VAR_NAME

        - None if the environment variable is not set
    """
    return os.environ.get(PRODUCTION_DIRECTORY_ENV_VAR_NAME)

def __fetch_production_dir_from_flag():
    """
    Looks for the command line flag --production_directory or -p
    Returns:
        The specified value if the flag is found
        None otherwise
    Raises:
        ProductionDirectoryNotSetError if the flag is found but no value was provided
    """
    index = None
    if PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME in sys.argv:
        index = sys.argv.index(PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME)
    elif PRODUCTION_DIRECTORY_CLI_FLAG_NAME in sys.argv:
        index = sys.argv.index(PRODUCTION_DIRECTORY_CLI_FLAG_NAME)

    if index is not None:
        try:
            return sys.argv[index+1]
        except IndexError:
            logging.error(f"""
        Oops, it looks like you used the production directory flag but you didn't specify a directory
        Make sure to pass a value after the --production_directory flag.
        Example:
            python -m smtag.predict.engine --demo {PRODUCTION_DIRECTORY_CLI_FLAG_NAME} "./rack"
            python -m smtag.predict.engine --demo {PRODUCTION_DIRECTORY_CLI_FLAG_SHORTNAME} "./rack"
        Your command was something like:
            {' '.join(sys.argv)}
            """)
            raise ProductionDirectoryNotSetError
    else:
        return None
