import os
import sys
import logging
from .errors import WorkingDirectoryNotSetError, WorkingDirectoryDoesNotExistError

DEFAULT_WORKING_DIRECTORY_NAME = "resources"
WORKING_DIRECTORY_ENV_VAR_NAME = "SMTAG_WORKING_DIRECTORY"
WORKING_DIRECTORY_CLI_FLAG_NAME = "--working_directory"
WORKING_DIRECTORY_CLI_FLAG_SHORTNAME = "-w"

def fetch_working_directory():
    """
    Returns the working directory path if specified by the user.
    If specified in different ways the order of precedence is:

        1. the CLI flag --working_directory
        2. the env variable SMTAG_WORKING_DIRECTORY
        3. the default folder

    Returns None otherwise
    """
    return (
        __fetch_working_dir_from_flag() or
        __fetch_working_dir_from_env() or
        __fetch_default_working_directory() or
        None
    )


def validated_working_directory(path):
    """
    Sanitzes the give path and makes sure it exists
    Receives:
        @param path: string specifying a path to the working directory

    Returns:
        The absolute path

    Raises:
        WorkingDirectoryDoesNotExistError if the path points to something that is not a directory
        or if it does not exist
    """
    if path is None:
        return None
    # convert to absolute path
    path =  os.path.abspath(path)
    # check that the directory exists and it is a directory
    if not os.path.isdir(path):
        raise WorkingDirectoryDoesNotExistError(path)
    return path

def __fetch_default_working_directory():
    """
    Returns the path to the default working directory to use if the user doesn't specify otherwise
    The default working directory name is configure via the constant `DEFAULT_WORKING_DIRECTORY_NAME`
    This directory can be found at the root of the project (e.g. where the setup.py file is)
    """
    return os.path.join(os.path.realpath(__file__), "..", "..", "..", "..", DEFAULT_WORKING_DIRECTORY_NAME)

def __fetch_working_dir_from_env():
    """
    Returns:
        - The value set in the environment variable with the name specified by
        the constant WORKING_DIRECTORY_ENV_VAR_NAME

        - None if the environment variable is not set
    """
    return os.environ.get(WORKING_DIRECTORY_ENV_VAR_NAME)

def __fetch_working_dir_from_flag():
    """
    Looks for the command line flag --working_directory or -w
    Returns:
        The specified value if the flag is found
        None otherwise
    Raises:
        WorkingDirectoryNotSetError if the flag is found but no value was provided
    """
    index = None
    if WORKING_DIRECTORY_CLI_FLAG_SHORTNAME in sys.argv:
        index = sys.argv.index(WORKING_DIRECTORY_CLI_FLAG_NAME)
    elif WORKING_DIRECTORY_CLI_FLAG_NAME in sys.argv:
        index = sys.argv.index(WORKING_DIRECTORY_CLI_FLAG_NAME)

    if index is not None:
        try:
            return sys.argv[index+1]
        except IndexError:
            logging.error(f"""
        Oops, it looks like you used the working directory flag but you didn't specify a directory
        Make sure to pass a value after the --working_directory flag.
        Example:
            python -m smtag.predict.engine --demo {WORKING_DIRECTORY_CLI_FLAG_NAME} "./resources"
            python -m smtag.predict.engine --demo {WORKING_DIRECTORY_CLI_FLAG_SHORTNAME} "./resources"
        Your command was something like:
            {' '.join(sys.argv)}
            """)
            raise WorkingDirectoryNotSetError
    else:
        return None
