# First time configuration
## With Python 3.6

Make sure to have `python3` available and run from to root of the project https://www.python.org/downloads/

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Alternative way with Anaconda
Install Anaconda https://www.anaconda.com/download/#macos

    # update anaconda
    ~/anaconda3/bin/conda update -n base conda

    # create a new virtual environment
    ~/anaconda3/bin/conda create --prefix ./.conda-venv python=3.6 anaconda

    echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
    source ~/.bash_profile

    # To activate this environment, use
    conda activate .conda-venv

    # To deactivate an active environment, use
    conda deactivate

# Contributing

* Remember to update the `requirements.txt` whenever you add a new python dependency to the project by running

    pip freeze > requirements.txt

* How to set up a breakpoint and debug in django

    import pdb; pdb.set_trace()

* Run the test suite

    python -m unittest discover
