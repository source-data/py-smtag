from setuptools import setup

setup(
    name='smtag_api',
    packages=['smtag_api'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)
