# stream2segment

A python project to download seismic waveforms related to events

This software has been createed on Mac OS El Capitan and tested under Ubuntu14.04. 
As of February 2016, the installation instructions refer to the latter. 

## Installation on Ubuntu14.04
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

# activate a virtual environment
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