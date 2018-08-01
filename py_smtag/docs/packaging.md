
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
