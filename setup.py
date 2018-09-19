import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="smtag",
    version="0.0.2.dev0",
    python_requires='>=3.6',
    author="Source Data",
    author_email="source_data@embo.org",
    description="SmartTag provides methods to tag text from figure legends based on the SourceData model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/source-data/py-smtag",
    packages=setuptools.find_packages(exclude=[
        'smtag.test',
    ]),
    py_modules=['smtag.command_line'],
    install_requires=[
        #'tensorflow==1.4', # needed for tensorboardX visualization
        'tensorboardX==1.2',
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
            'smtag-neo2xml=smtag.command_line:neo2xml',
            'smtag-convert2th=smtag.command_line:convert2th',
            'smtag-meta=smtag.command_line:meta',
            'smtag-predict=smtag.command_line:predict',
            'smtag=smtag.command_line:about',
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
