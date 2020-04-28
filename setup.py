import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="smtag",
    version="2.0.2",
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
        #'tensorflow==1.8.0', # dgx
        'tensorflow==1.14.0', # mac needed for tensorboardX visualization
        'tensorboardX',
        'docopt',
        'torch==1.0.1',
        'torchvision==0.2.2.post3',
        'neo4jrestclient==2.1.1',
        'nltk',
        'google-cloud-vision==0.34',
        'google-auth==1.5.1',
        'opencv-python',
        'scikit-learn',
        'python-dotenv==0.10.1',
        'google-cloud-vision==0.34',
    ],
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            'smtag-neo2xml=smtag.command_line:neo2xml',
            'smtag-ocr=smtag.command_line:ocr',
            'smtag-viz=smtag.command_line:viz',
            'smtag-convert2th=smtag.command_line:convert2th',
            'smtag-meta=smtag.command_line:meta',
            'smtag-eval=smtag.command_line:eval',
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
