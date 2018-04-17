# First time configuration
## With Python 3.6

Make sure to have `python3` available and run from to root of the project https://www.python.org/downloads/

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Copy `.env.example` to `.env` and update accordingly. Defaults may work but pay special attentionn to `SMTAG_LUA_CLI_PATH` and `TORCH_PATH`, which will define where to find both `smtagCli.lua` and `th` (run `which th`) files. Consider using absolute paths instead of relatives as in the example file.

# Contributing

* Run the web server

In bash

    export FLASK_APP=smtag_api
    export FLASK_DEBUG=true
    pip install --editable .
    flask run

In fish

    set -x FLASK_APP smtag_api
    set -x FLASK_DEBUG true
    pip install --editable .
    flask run

* Remember to update the `requirements.txt` whenever you add a new python dependency to the project by running

    ```
    pip freeze > requirements.txt
    ```

* How to set up a breakpoint to get an interactive debugger console in Python

    ```
    import pdb; pdb.set_trace()
    ```

* Run the test suite

    ```
    python setup.py test
    ```
