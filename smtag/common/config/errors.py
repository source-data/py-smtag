import logging

class ConfigError(Exception):
    pass

class ProdDirNotFoundError(ConfigError):
    def __init__(self, path):
        super(ProdDirNotFoundError, self).__init__()
        logging.error(f"""
            #######################################################################################################
            #
            ERROR: PROD DIR NOT FOUND

            The specified working directory does not exist:

                {path}

            #
            #######################################################################################################
        """)
