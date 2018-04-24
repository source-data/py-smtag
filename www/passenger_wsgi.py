import sys, os
from smtag_api import app as application

# root of the py-smtag git repo
HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

VENV = HOME + '/shared/venv'
PYTHON_BIN = VENV + '/bin/python3'

if sys.executable != PYTHON_BIN:
    os.execl(PYTHON_BIN, PYTHON_BIN, *sys.argv)

sys.path.insert(0, '{v}/lib/python3.6/site-packages'.format(v=VENV))
