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

# Deploying

[Capistrano](http://capistranorb.com/) is included to atuomate the automate deployment of this web project.

Two different environments are configured, `staging` (meant for internal testing) and `production`. To deploy your freshly pushed code to any of these simply run form your local computer

    bundle install # only the first time
    bundle exec cap staging deploy # deploys to staging
    bundle exec cap production deploy # deploys to production

Keep in mind that:

* The computer where you deploy from (typically your own laptop) needs to have ssh access to the server you are trying to deploy to.
* `production` always deploys the `master` branch of the git repository
* but `staging` deploys the branch that you are currently on

The specifics on how to setup your Linux boxes, choose and configure your http servers or configure the deployment scripts are outside of the scope of this document.

