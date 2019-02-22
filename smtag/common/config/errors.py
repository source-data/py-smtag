import logging
class ConfigError(Exception):
    pass
class WorkingDirectoryNotSetError(ConfigError):
    def __init__(self):
        super(WorkingDirectoryNotSetError, self).__init__()
        logging.error("""
            #######################################################################################################
            #
            ERROR: WORKING DIRECTORY NOT SET

            You tried to use `smtag.config.working_directory` either directly or inderectly, but there
            is no `working_directory` defined.

            In order to configure one use one of this options, sorted by lesser precedence:
            1) Set the environment variable `SMTAG_WORKING_DIRECTORY`
               Examples:
                SMTAG_WORKING_DIRECTORY='/absolute/path/to/resources_folder' python -m smtag.predict.egine --demo
                SMTAG_WORKING_DIRECTORY='./relative/path/to/resources_folder' python -m smtag.predict.egine --demo
            2) Set the `--working-directory` flag or its abbreviated version `-w`
               Example:
                python -m smtag.predict.egine --demo --working-directory "/absolute/path/to/resources_folder"
                python -m smtag.predict.egine --demo -w "./relative/path/to/resources_folder"
            3) Set the `smtag.config.working_directory` programatically
               Example:
                import smtag
                smtag.config.working_directory = "/absolute/path/to/resources_folder"
                smtag.config.working_directory = "./relative/path/to/resources_folder"
            #
            #######################################################################################################
            """)
class WorkingDirectoryDoesNotExistError(ConfigError):
    def __init__(self, path):
        super(WorkingDirectoryDoesNotExistError, self).__init__()
        logging.error(f"""
            #######################################################################################################
            #
            ERROR: WORKING DIRECTORY DOES NOT EXIST

            The specified working directory does not exist:

                {path}


            Please review carefully:
            1) Your env variable `SMTAG_WORKING_DIRECTORY`
            2) The command line flag `--working-directory` or its short alias `-w`
            3) If you are programatically setting the working_directory, for example:
                import smtag
                smtag.config.working_directory = "/absolute/path/to/resources_folder"
            #
            #######################################################################################################
        """)
