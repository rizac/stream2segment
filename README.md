# stream2segment

A python project to download seismic waveforms related to events

This software has been createed on Mac OS El Capitan and tested under Ubuntu14.04. 
As of February 2016, the installation instructions refer to the latter. 

## Installation on Ubuntu14.04

### Short installation (not tested, it's a summary of the more verbose one described below)

Prerequisites for all packages to work:
```
sudo apt-get update
sudo apt-get upgrade gcc
sudo apt-get install git python-pip python2.7-dev libpng-dev libfreetype6-dev \
	build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev
```

*
For a reference on numpy problems later (like "Cannot compile 'Python.h'") then
see http://stackoverflow.com/questions/18785063/install-numpy-in-python-virtualenv

Additional libraries info:
libpng-dev libfreetype6-dev are required for matplotlib
(see http://stackoverflow.com/questions/25593512/cant-install-matplotlib-using-pip
and http://stackoverflow.com/questions/28914202/pip-install-matplotlib-fails-cannot-build-package-freetype-python-setup-py-e)

build-essential gfortran libatlas-base-dev are required for scipy
(see http://stackoverflow.com/questions/2213551/installing-scipy-with-pip/3865521#3865521)

libxml2-dev libxslt-dev are required for lxml
(see here: http://lxml.de/installation.html)
*

We strongly recomend to use python virtual environment. Install python virtual environment
```
sudo pip install virtualenv
```

Clone repository to a specific folder $F
```
git clone https://github.com/rizac/stream2segment.git
```
Activate virtualenv, move to package folder:
```
cd stream2segment
```
Make virtual environment in an stream2segment/env directory (env is a convention, but it's ignored by git commits so keep it)
```
virtualenv env
```
and activate it: (THIS TO  BE DONE EACH TIME THE SCRIPT IS RUN)
```
source env/bin/activate
```

*
To check you are in the right env, type:
which pip
you should see it's pointing inside the env folder
*

Install numpy, to be done first of all
```
pip install numpy==1.10.4
```

Install the current package
```
pip install -e .
```

Copy default config to a config file (not included in git commit):
```
cp config.example.yaml config.yaml
```
and edit it if you whish

### Usage
Move to the stream2segment folder, activate the virtual environment
```
source env/bin/activate
```
and run
```
stream2segment
```





### Long installation (true report of the tested installation)

```
# install git if not installed
sudo apt-get install git

# clone repository to a specific folder $F
git clone https://github.com/rizac/stream2segment.git
# (now you have $F/stream2segment)

# is pip installed? NO? then
sudo apt-get install python-pip

# in case of error:
sudo apt-get update
# and repeat python-pip install

# install python virtual environment
sudo pip install virtualenv

# activate virtualenv:
cd stream2segment
# make virtual environment in an $F/env directory (env is a convention, but it's ignored by git commits so keep it)
virtualenv env
# activate it: (THIS TO  BE DONE EACH TIME THE SCRIPT IS RUN)
source env/bin/activate

#to check you are in the right env, type:
which pip
# you should see it's pointing inside the env folder

# install numpy
pip install numpy==1.10.4
# better than 'pip install numpy'

# problems like "Cannot compile 'Python.h'"? then:
sudo apt-get update
sudo apt-get upgrade gcc
sudo apt-get install python2.7-dev
# see http://stackoverflow.com/questions/18785063/install-numpy-in-python-virtualenv

# now install the package
pip install -e .

# problems with matplotlib? read what is missing and try googling to install it. Most probably you need to run:
# see http://stackoverflow.com/questions/25593512/cant-install-matplotlib-using-pip
# and http://stackoverflow.com/questions/28914202/pip-install-matplotlib-fails-cannot-build-package-freetype-python-setup-py-e
sudo apt-get install libpng-dev libfreetype6-dev

# problems installing scipy? see http://stackoverflow.com/questions/2213551/installing-scipy-with-pip/3865521#3865521
sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev

# problems installing lxml?
# see here: http://lxml.de/installation.html
sudo apt-get install libxml2-dev libxslt-dev python-dev

# copy default config to a config file (not included in git commit):
cp config.example.yaml config.yaml

#run script
stream2segment
```

