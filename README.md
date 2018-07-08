# First time configuration
## With Python 3.6

Make sure to have `python3` available and run from to root of the project https://www.python.org/downloads/

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install --upgrade tensorflow # see doc/tensorboard.md

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


# Training: Command Line Interface Usage

Training of new models is managed via `meta` in the `train` module. Run the help command to get a list of options

    $ python -m train.meta --help

    usage: meta.py [-h] [-f FILE] [-E EPOCHS] [-Z MINIBATCH_SIZE]
                [-R  LEARNING_RATE] [-V VALIDATION_FRACTION]
                [-o OUTPUT_FEATURES] [-i FEATURES_AS_INPUT]
                [-a OVERLAP_FEATURES] [-c COLLAPSED_FEATURES] [-n NF_TABLE]
                [-k KERNEL_TABLE] [-p POOL_TABLE]

    Top level module to manage training.

    optional arguments:
    -h, --help            show this help message and exit
    -f FILE, --file FILE  Namebase of dataset to import (default:
                            test_entities_train)
    -E EPOCHS, --epochs EPOCHS
                            Number of training epochs. (default: 120)
    -Z MINIBATCH_SIZE, --minibatch_size MINIBATCH_SIZE
                            Minibatch size. (default: 128)
    -R  LEARNING_RATE, --learning_rate LEARNING_RATE
                            Learning rate. (default: 0.001)
    -V VALIDATION_FRACTION, --validation_fraction VALIDATION_FRACTION
                            Fraction of the dataset that should be used as
                            validation set during training. (default: 0.2)
    -o OUTPUT_FEATURES, --output_features OUTPUT_FEATURES
                            Selected output features (use quotes if comma+space
                            delimited). (default: geneprod)
    -i FEATURES_AS_INPUT, --features_as_input FEATURES_AS_INPUT
                            Features that should be added to the input (use quotes
                            if comma+space delimited). (default: )
    -a OVERLAP_FEATURES, --overlap_features OVERLAP_FEATURES
                            Features that should be combined by intersecting them
                            (equivalent to AND operation) (use quotes if
                            comma+space delimited). (default: )
    -c COLLAPSED_FEATURES, --collapsed_features COLLAPSED_FEATURES
                            Features that should be collapsed into a single one
                            (equivalent to OR operation) (use quotes if
                            comma+space delimited). (default: )
    -n NF_TABLE, --nf_table NF_TABLE
                            Number of features in each hidden super-layer.
                            (default: [8, 8, 8])
    -k KERNEL_TABLE, --kernel_table KERNEL_TABLE
                            Convolution kernel for each hidden layer. (default:
                            [6, 6, 6])
    -p POOL_TABLE, --pool_table POOL_TABLE
                            Pooling for each hidden layer (use quotes if
                            comma+space delimited). (default: [2, 2, 2])

Try demo training:

    $ python -m train.meta


# Semantic tagging: SmartTag engine

The SmartTag engine is located in the `predict` module. It loads automatically models from `rack/` and processes text to segment it, tag terms and assign context-dependent semantic categories.

    $ python -m predict.engine --help
    SmartTag semantic tagging engine.

    Usage:
    engine.py [-D -d -m <str> -t <str> -f <str>]

    Options:

    -m <str>, --method <str>                Method to call (smtag|tag|entity|panelize) [default: smtag]
    -t <str>, --text <str>                  Text input in unicode [default: Fluorescence microcopy images of GFP-Atg5 in fibroblasts from Creb1-/- mice after bafilomycin treatment.].
    -f <str>, --format <str>                Format of the output [default: xml]
    -D, --debug                             Debug mode to see the successive processing steps in the engine.
    -d, --demo                              Demo with a long sample.

    To visualize the engine's processing steps in action and to debug, use:

    $ python -m predict.engine -D


# Contributing

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
    python -m unittest discover
    ```
