# <img align="left" height="30" src="https://www.gfz-potsdam.de/fileadmin/gfz/medien_kommunikation/Infothek/Mediathek/Bilder/GFZ/GFZ_Logo/GFZ-Logo_eng_RGB.svg"> Stream2segment <img align="right" height="50" src="https://www.gfz-potsdam.de/fileadmin/gfz/GFZ_Wortmarke_SVG_klein_en_edit.svg">

A Python project to download, process and visualize event-based seismic waveform segments.

The key aspects with respect to widely-used similar applications are:

* A database storage (sqlite or postgres) for downloaded data (and metadata)
* A highly customizable processing module which can, e.g., create csv file outputs or store
  on the local file system the processed waveform segments.
* A visualization tool to show downloded and optionally customized processed segments in a web browser by means
  of Python we-app and Javascript plotting libraries. The user can also set class labels 
  to make the GUI a hand-labelling tool for supervised classification problems, or to simply annotate special segments
* A highly efficient, easy-to-use selection of segments for filtering data for processing and/or visualization. The selection can be performed on all segments metadata, it exploits the efficiency of SQL 'select' syntax and its simplified for non-experienced user with a documented and simplified custom syntax.

**Citation (Software):**
> Zaccarelli, Riccardo (2018): Stream2segment: a tool to download, process and visualize event-based seismic waveform data. V. 2.7.3. GFZ Data Services.

[http://doi.org/10.5880/GFZ.2.4.2019.002](http://doi.org/10.5880/GFZ.2.4.2019.002)


**Citation (Research article):**:
> Riccardo Zaccarelli, Dino Bindi, Angelo Strollo, Javier Quinteros and Fabrice Cotton. Stream2segment: An Open‐Source Tool for Downloading, Processing, and Visualizing Massive Event‐Based Seismic Waveform Datasets. *Seismological Research Letters* (2019)

[https://doi.org/10.1785/0220180314](https://doi.org/10.1785/0220180314)


## Usage

Please refer to the [github documentation](https://github.com/rizac/stream2segment/wiki)


## Installation

This program has been installed and tested on Ubuntu14.04, Ubuntu16.04 and Mac OSX El Capitan

### Prerequisites

This is a set of system packages which are necessary to run the program on Ubuntu.
On Mac OsX El Capitan, we **did not experience the need of these packages** as they are probably
pre-installed, so you can try to skip this section in the first place and get back here just in case.

#### Ubuntu (tested on 14.04 and 16.04)

We suggest to upgrade `gcc` first:
```
sudo apt-get update
sudo apt-get upgrade gcc
```
<!--
The following system packages are required: `git python-pip python2.7-dev libpng-dev libfreetype6-dev 
build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev`
-->

You can skip installing the system packages and get back here in case of problems, or choose to be
(almost) sure (see also [Installation Notes](#installation-notes)):

*NOTE* Replace `python2.7-dev` with `python3-dev` and `python-pip` with `python3-pip` if you want to use the tool under  python3

```
sudo apt-get update
sudo apt-get install git python-pip python2.7-dev libpng-dev libfreetype6-dev \
	build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev python-tk
```

Another option is to install **really** required packages:
```
sudo apt-get update
sudo apt-get install git python-pip python2.7-dev  # python 2
sudo apt-get install git python3-pip python3-dev  # python 3
```

and get back here in case of problems

<!--
#### Ubuntu16.04 and Python3.5+

We suggest to upgrade `gcc` first (this has been proved necessary because some tests failed before
upgrading and did not afterwards):
```
sudo apt-get update
sudo apt-get upgrade gcc
```
You can skip installing the system packages and get back here in case of problems, or choose to be
(almost) sure (see also [Installation Notes](#installation-notes)):
```
sudo apt-get update
sudo apt-get install git python3-pip wheel
```
-->

### Cloning repository

Git-clone (basically: download) this repository to a specific folder of your choice:
```
git clone https://github.com/rizac/stream2segment.git
```
and move into package folder:
```
cd stream2segment
```

### Install and activate python virtualenv

(If using Anaconda, skip this section and go to the next one)

We strongly recomend to use python virtual environment, because by isolating all python packages we are about to install, we won't create conflicts with already installed packages.

#### Installation (all versions)
To install python virtual environment either use [Virtualenvwrapper][http://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation] or the more low-level approach `virtualenv`:
```
sudo pip install virtualenv
```
Make virtual environment in an stream2segment/env directory (env is a convention, but it's ignored by git commits so better keep it)
 ```
virtualenv env
 ```
(on ubuntu 16.04, we got the message 'virtualenv: Command not found.'. We just typed: `/usr/local/bin/virtualenv env`)
and activate it:

#### Installation (alternative for python3)
Python 3 has a built-in support for virtual environments - venv. It might be better to use that instead. To install
```
sudo apt-get install python3-venv
```
Make virtual environment in an stream2segment/env directory (env is a convention, but it's ignored by git commits so better keep it)
```
python3 -m venv ./env
```

#### Activation (any python version)
 ```
 source env/bin/activate
 ```
or `source env/bin/activate.csh` (depending on your shell)

> <sub>Activation needs to be done __each time__ we will run the program.</sub>
> <sub>To check you are in the right env, type: `which pip` and you should see it's pointing inside the env folder</sub>


### Install and activate python virtualenv (Anaconda)

(Thanks to JessieMyr who wrote this for us **disclaimer: the lines below might be outdated. Please refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for details**)

Create a virtual environment for your project

  - In the terminal client enter the following where yourenvname (like « env ») is the name you want to call your environment, and replace x.x with the Python version you wish to use. (To see a list of available python versions first, type conda search "^python$" and press enter.)
	```
	Conda create –n yourenvname python=x.x anaconda
	```
  - Press « y » to proceed

Activate your virtual environment

  - ```$source activate env```
  - To deactivate this environment, use ```$source deactivate```


### Install and config packages

**From now on you are supposed to be in your activated python virtualenv**

Run `pip freeze`. If you get the message 'You are using pip version ..., however version ... is available.'
then execute:
```
pip install --upgrade pip
```

#### Install using pre-built scripts ...

Run `./installme` or `./installme-dev` (the latter if you want to contribute and/or
run tests to check if the program will likely work in your system)

#### ... or the old (longer) way:

Install via requirements file:
```
pip install ./requirements.txt
```
Alternatively, if you want to run tests (recommended):
```
pip install ./requirements.dev.txt
```
Install the current package
```
pip install -e .
```

If you wish to use the program within Jupyter notebooks, jupyter is not included
in the dependencies. Thus
```
pip install jupyter
```

The program is now installed. To double check the program functionalities, we suggest to run tests (see below) and report the problem in case of failure 

### Runt tests

Stream2segment has been highly tested (current test coverage is above 90%).
Continuous integration tests are not yet in place, the program has been tested
on Python version from 3.5+ and above 2.7. **Note (summer 2019): we will discontinue support for
Python 2.7 very soon**

To execute tests, move in the project directory and run the command:
```
pytest -xvvv -W ignore --dburl <additional db url> ./tests/
```
or, if 'pytest not found' message appears:
```
python -m pytest ...
```
Run tests tracking coverage:
```
pytest -xvvv -W ignore --dburl <additional db url> --cov=./stream2segment --cov-report=html ./tests/
```
(short explanation: `-x`: stop at first error, `-vvv`: increase verbosity, `-W ignore`: do not print Python warnings issued during tests. This might be very useful because when tests fail, these warnings might hide in the terminal the important failure messages. Note that other options of `-W`, e.g. a very useful "print only first warning by type", seem not to work with current pytest version)

Tests are time consuming (some minutes currently) and you should see a message with no errors, such as
`"===== ### passed in ### seconds ======"`

The database used for testing will be an sqlite database. If you want to provide other database type their urls via the option ```--dburl``` (you can type it multiple times and all tests requiring a database will be run with all provided database urls).
Example: if you have postgres installed with an *already created* database named ```s2s_test```, run:
```
pytest -xvvv -W ignore --dburl postgresql://<user>:<password>@localhost/<dbname> ./tests/
```
(the data on any given database will be overwritten if the database is not empty)


## Installation Notes:

- If you see (we experienced this while running tests, thus we can guess you should see it whenever accessing the program
  for the first time):
  ```
  This system supports the C.UTF-8 locale which is recommended.
  You might be able to resolve your issue by exporting the
  following environment variables:

    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
  ```
  Then edit your `~/.profile` (or `~/.bash_profile` on Mac) and put the two lines starting with 'export',
  and execute `source ~/.profile` (`source ~/.bash_profile` on Mac) and re-execute the program.  

- On Ubuntu 12.10, there might be problems with libxml (`version libxml2_2.9.0' not found`). 
Move the file or create a link in the proper folder. The problem has been solved looking at
http://phersung.blogspot.de/2013/06/how-to-compile-libxml2-for-lxml-python.html

- On Ubuntu 14.04 
All following issues should be solved by following the instructions in the section [Prerequisites](#prerequisites). However:
 - For numpy installation problems (such as `Cannot compile 'Python.h'`) , the fix
has been to update gcc and install python2.7-dev: 
	```sudo apt-get update
	sudo apt-get upgrade gcc
	sudo apt-get install python2.7-dev```
	For details see http://stackoverflow.com/questions/18785063/install-numpy-in-python-virtualenv
 - For scipy problems, `build-essential gfortran libatlas-base-dev` are required for scipy (see http://stackoverflow.com/questions/2213551/installing-scipy-with-pip/3865521#3865521)
 - For lxml problems, `libxml2-dev libxslt-dev` are required (see here: http://lxml.de/installation.html)

 - For matplotlib problems (matplotlib is not used by the program but from imported libraries), `libpng-dev libfreetype6-dev` are required (see http://stackoverflow.com/questions/25593512/cant-install-matplotlib-using-pip and http://stackoverflow.com/questions/28914202/pip-install-matplotlib-fails-cannot-build-package-freetype-python-setup-py-e)

## Misc:

### sqlitebrowser
The program saves data on a sql database (tested with postresql and sqlite). If sqlite is used as database, to visualize the sqlite content you can download sqlitebrowser (http://sqlitebrowser.org/). The installation on Mac is straightforward (use brew cask or go to the link above) whereas on Ubuntu can be done as follows:
```
sudo add-apt-repository ppa:linuxgndu/sqlitebrowser
sudo apt-get install sqlitebrowser
```

### matplotlibrc

A `matplotlibrc` file is included in the main root package. As said, matplotlib is not used by the program
but from imported libraries, The included file sets the backend to 'Agg' so that we hide the "Turning interactive mode on" message (for Mac users)


## Developer(s) notes:

- The program can be also installed via the usual way:
  ```
  pip install -e .
  ```
  which means that the latest library dependancies will
  be downloaded. This however is not as safe as using `requirements.txt` because tests passed with the
  specific versions implemented therein. For developers whishing to 'upgrade' to new libraries, thus
  changing requirements.txt (and making pip install most likely work for a relatively medium period of time),
  the procedure is:
  ```
	pip install -e .
	pip freeze > ./requirements.tmp
	pip install -e .[dev,test]
	pip freeze > ./requirements.dev.tmp
  ```
  Run tests (see above), if everything is fine you can replace old `requirements.txt` and
  `requirements.dev.txt` with the `.tmp` file created
