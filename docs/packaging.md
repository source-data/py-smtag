
# Useful links:

https://packaging.python.org/tutorials/packaging-projects/
https://packaging.python.org/guides/distributing-packages-using-setuptools/
http://python-packaging.readthedocs.io/en/latest/
http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html

you need to have a setup.py

python setup.py sdist
pip install wheel
python setup.py bdist_wheel
python setup.py install
python setup.py develop





python -m py_smtag.train.meta -Z10
python -m py_smtag.datagen.sdgraph2th -l20 -L123 -y gene,protein -X1 -W -f test_entities
python -m py_smtag.predict.engine -d
python -m py_smtag.predict.engine -D
python -m py_smtag.predict.engine -D -t "Alex is the greatest mutant."



# How to install the package if it is not release to PyPI yest

```
# install with pip from the github repo the branch `tldev`
pip install git+ssh://git@github.com/source-data/py-smtag.git@tldev

# install with pip from a repo on your local drive the branch `tldev`
pip install git+file:///Users/alejandroriera/dev/py-smtag@tldev

# install with pip from a local folder (this is different from local git repo in the sense that you don't have to commit your changes)
pip install -e /Users/alejandroriera/dev/py-smtag
```
