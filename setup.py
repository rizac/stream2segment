"""A setuptools based setup module.
Taken from:
https://github.com/pypa/sampleproject/blob/master/setup.py

See also:
http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/

Additional links:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from __future__ import print_function

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
version = ""
with open(path.join(here, 'stream2segment', 'resources', 'program_version')) as version_file:
    version = version_file.read().strip()

setup(
    name='stream2segment',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='A python project to download, process and visualize event-based seismic waveforms',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/rizac/stream2segment',

    # Author details
    author='riccardo zaccarelli',
    author_email='rizac@gfz-potsdam.de',  # FIXME: what to provide?

    # Choose your license
    license='GNU',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU License',


        # Specify the Python versions you support here.
        # 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

    # What does your project relate to?
    keywords='download seismic waveforms related to events',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'htmlcov']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For info see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['PyYAML>=3.12',
                      'numpy>=1.13.1',
                      'tables>=3.5.2',
                      'pandas>=0.20.3',
                      'obspy>=1.0.3',
                      'Flask>=0.12.3',
                      'psycopg2>=2.7.3.1',
                      'psutil>=5.3.1',
                      'SQLAlchemy>=1.1.14,<1.4',
                      'click>=6.7'
                      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]  (pip install -e ".[dev,test]" in zsh)
    extras_require={
        # use latest versions. Without boundaries
        'dev': ['pep8>=1.7.0',
                'pylint>=1.7.2',
                'pytest>=3.2.2',
                'pytest-cov>=2.5.1',
                'pytest-mock>=1.6.2'],
        'jupyter': ['jupyter>=1.0.0']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #
    # package_data={
    #    'sample': ['package_data.dat'],
    # },

    # make the installation process copy also the package data (see MANIFEST.in)
    # for info see https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'stream2segment=stream2segment.cli:cli',
            's2s=stream2segment.cli:cli',
        ],
    },
)
