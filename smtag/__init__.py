name = "smtag"

################################################################################
# Load .env file
#
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

################################################################################
# Initialize smtag config
#
from .common.config.configuration import Config
config = Config()
