# stream2segment

A python project to download seismic waveforms related to events featuring several pre- and
post- download processing utilities

## Installation

This program has been installed and tested on Ubuntu14.04, Ubuntu16.04 and Mac OSX El Capitan.
The installation instructions below refer to the former (Ubuntu). In principle, Mac users can
safely follow the same instructions below.

### Prerequisites

This is a set of system packages which are necessary to run the program on Ubuntu.
On Mac OsX El Capitan, we **did not experience the need of these packages** as they are probably
pre-installed, so you can try to skip this section in the first place and get back here just in case.

#### Ubuntu14.04 and Python2.7+

We suggest to upgrade `gcc` first:
```
sudo apt-get update
sudo apt-get upgrade gcc
```
The following system packages are required: `git python-pip python2.7-dev libpng-dev libfreetype6-dev 
build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev`
You can skip installing the system packages and get back here in case of problems, or choose to be
(almost) sure (see also [Installation Notes](#installation-notes)):
```
sudo apt-get update
sudo apt-get install git python-pip python2.7-dev libpng-dev libfreetype6-dev \
	build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev python-tk
```

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
sudo apt-get install git python3-pip
```

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

We strongly recomend to use python virtual environment, because by isolating all python packages we are about to install, we won't create conflicts with already installed packages.

#### Installation (all versions)
To install python virtual environment either use [Virtualenvwrapper][http://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation] or the more low-level approach `virtualenv`:
```
sudo pip install virtualenv
```
Make virtual environment in an stream2segment/env directory (env is a convention, but it's ignored by git commits so keep it)
 ```
virtualenv env
 ```
(on ubuntu 16.04, we got the message 'virtualenv: Command not found.'. We just typed: `/usr/local/bin/virtualenv env`)
and activate it:

#### Installation (alternative for python3)
Python 3 has a built-in support for virtual environments - venv. It might be better to use that instead. To install
```
sudo apt-get install python3-venv wheel
```
Make virtual environment in an stream2segment/env directory (env is a convention, but it's ignored by git commits so keep it)
```
python3 -m venv ./env
```
You might 
#### Activation (any python version)
 ```
 source env/bin/activate
 ```
or `source env/bin/activate.csh` (depending on your shell)

> <sub>Activation needs to be done __each time__ we will run the program. See section [Usage](#usage) below</sub>
> <sub>Check: To check you are in the right env, type: `which pip` and you should see it's pointing inside the env folder</sub>


#### Virtualenv with anaconda

(Thanks to JessieMyr who wrote this for us)

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

Now you are supposed to be in your python virtualenv

Run `pip freeze`. If you get the message 'You are using pip version ..., however version ... is available.'
then execute:
```
pip install --upgrade pip
```

#### Using pre-built scripts ...

Run `./installme` or `./installme-dev` (the latter if you want to run tests, recommended)

#### ... or the old (longer) way:

Install via requirements file:
```
pip install ./requirements.txt
```
Or, if you want to run tests (recommended):
```
pip install ./requirements.dev.txt
```
Install the current package
```
pip install -e .
```

### Runt tests

To run tests, move in the project directory and run the command:
```
py.test ./tests/ --ignore=./tests/skip --cov=./stream2segment
```
or (if 'pytest not found' message appears):
```
python -m pytest ./tests/ --ignore=./tests/skip --cov=./stream2segment
```
(you should see a message with no errors, such as "===== 8 passed in 1.30 seconds ======")

## Usage

(more on this upcoming...)

Move (`cd` on a terminal) to the stream2segment folder. If you installed and activated a python virtual environment during installation (hopefully you did), **activate the virtual environment first**:
```
source env/bin/activate
```

> <sub>When you're finished, type `deactivate` on the terminal to deactivate the current pythoin virtual environment and return to the global system defined Python</sub>

Type:
```
s2s t --help
```
and check how to create download and processing template files. Once done, type
```
s2s d --help
```
to see how to execute the download. Edit the download yaml file created with `s2s t --help` and run the
download. If a postgres database is used, setup the database first.

For processing, type:
```
s2s p --help
```
or alternatively, for visualizing the downloaded data, type:
```
s2s v --help
```
Edit the processing files (yaml and python) created with `s2s t --help` and run the processing
or the GUI

## Installation Notes:

- On Ubuntu 12.10, there might be problems with libxml (`version libxml2_2.9.0' not found`). 
Move the file or create a link in the proper folder. The problem has been solved looking at
http://phersung.blogspot.de/2013/06/how-to-compile-libxml2-for-lxml-python.html

- On Ubuntu 14.04 
All following issues should be solved by following the instructions in the section [Prerequisites](#prerequisites). However,
 - For numpy installation problems (such as `Cannot compile 'Python.h'`) , the fix
has been to update gcc and install python2.7-dev: 
	```sudo apt-get update
	sudo apt-get upgrade gcc
	sudo apt-get install python2.7-dev```
	For details see http://stackoverflow.com/questions/18785063/install-numpy-in-python-virtualenv
 - For scipy problems, `build-essential gfortran libatlas-base-dev` are required for scipy (see http://stackoverflow.com/questions/2213551/installing-scipy-with-pip/3865521#3865521)
 - For lxml problems, `libxml2-dev libxslt-dev` are required (see here: http://lxml.de/installation.html)

(We do not use anymore matplotlib for responsive GUIs, you can skip the lines below)
 - ~~For matplotlib problems, `libpng-dev libfreetype6-dev` are required (see http://stackoverflow.com/questions/25593512/cant-install-matplotlib-using-pip and http://stackoverflow.com/questions/28914202/pip-install-matplotlib-fails-cannot-build-package-freetype-python-setup-py-e)~~
 - ~~For matplotlib problems, if running `stream2segment --gui` (after downloading some data), you get `ImportError: cannot import name _tkagg` you should install python-tk: ```apt-get install python-tk`` (see http://stackoverflow.com/questions/4783810/install-tkinter-for-python)~~

## Misc:

### sqlitebrowser
The program saves data on a sql database (tested with postresql and sqlite). If sqlite is used as database, to visualize the sqlite content you can download sqlitebrowser (http://sqlitebrowser.org/). The installation on Mac is straightforward (use brew cask or go to the link above) whereas on Ubuntu can be done as follows:
```
sudo add-apt-repository ppa:linuxgndu/sqlitebrowser
sudo apt-get install sqlitebrowser
```

### ~~matplotlibrc~~

A `matplotlibrc` file is included in the main root package. It sets the backend to 'TkAgg' so that
we hide the "Turning interactive mode on" message (for Mac users) when importing packages
requiring matplotlib (the program does not use anymore `matplotlib` for responsive GUIs
as we moved to more robust and more powerful web interfaces with Flask).
