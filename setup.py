import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_smtag",
    version="0.0.1.dev0",
    python_requires='>=3.6',
    author="Source Data",
    author_email="source_data@embo.org",
    description="SmartTag provides methods to tag text from figure legends based on the SourceData model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/source-data/py-smtag",
    packages=setuptools.find_packages(exclude=[
        'py_smtag.test',
    ]),
    py_modules=['py_smtag.command_line'],
    install_requires=[
        'tensorflow==1.4', # needed for tensorboardX visualization
        'docopt==0.6.2',
        'numpy==1.14.2',
        'Pillow==4.3.0', #5.1.0 requires Mac OS 10.12; torchvision require at least 4.1.1; GPU EC2 AMI needs >4.3.0
        'PyYAML==3.12',
        'six==1.10.0', # GPU EC2 AMI need 1.10
        'torch==0.4.0',
        'torchvision==0.2.0',
        'neo4jrestclient==2.1.1',
        'nltk==3.2.4',
    ],
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            'smtag-graph2th=py_smtag.command_line:graph2th',
            'smtag-predict=py_smtag.command_line:predict',
            'smtag-meta=py_smtag.command_line:meta',
            'smtag=py_smtag.command_line:about',
        ],
    },
    # keywords="",
    classifiers=(
        # full list: https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries",
    ),
)
