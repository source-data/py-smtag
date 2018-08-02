# How to run meta, engine and others from the command line
The new package creates different command line entry points (see `setup.py`),
but these will only be accessible after running `python setup.py install`.

    smtag-graph2th -l20 -L123 -y gene,protein -X1 -W -f test_entities
    smtag-predict -D -t "Alex is the greatest mutant."
    smtag-meta -Z10

However after every change in the code you need to install it again, changes
don't get automatically updated. That's why During development it can be easier
to simply load your module from the command line, similarly to what we did
before:

    python -m smtag.train.meta -Z10
    python -m smtag.datagen.sdgraph2th -l20 -L123 -y gene,protein -X1 -W -f test_entities
    python -m smtag.predict.engine -D -t "Alex is the greatest mutant."


# How to install the package if it is not released to PyPI yet

Once we release a package to world by publishing it in [PyPI](https://pypi.org/)
we will be able to install it with something like `pip install pysmtag`.

Until then, if you want to install it in your computer (or a remote server) you
can do it in 3 different ways:

```
# install with pip from the github repo the branch `tldev`
pip install git+ssh://git@github.com/source-data/py-smtag.git@tldev

# install with pip from a repo on your local drive the branch `tldev`
pip install git+file:///Users/alejandroriera/dev/py-smtag@tldev

# install with pip from a local folder (this is different from local git repo
# in the sense that you don't have to commit your changes)
pip install -e /Users/alejandroriera/dev/py-smtag
```

# Read more:

* https://packaging.python.org/tutorials/packaging-projects/
* https://packaging.python.org/guides/distributing-packages-using-setuptools/
* http://python-packaging.readthedocs.io/en/latest/
* http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html

# Useful commands

    python setup.py sdist
    pip install wheel
    python setup.py bdist_wheel
    python setup.py install
    python setup.py develop


