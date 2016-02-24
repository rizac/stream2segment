# stream2segment

A python project to download seismic waveforms related to events

## Installation (tested on Ubuntu14.04 - Ubuntu 12.10)

### Installation

#### Prerequisites
(the following lines might be skipped if everything is correctly already setup.
If you do, in case of problems refer to the Notes below)

```
sudo apt-get update
sudo apt-get upgrade gcc
sudo apt-get install git python-pip python2.7-dev libpng-dev libfreetype6-dev \
	build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev
```

#### Python tools and dependances
We strongly recomend to use python virtual environment. Install python virtual environment
```
sudo pip install virtualenv
```

Clone repository to a specific folder of your choice (`cd` into that folder first)
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

*Check: To check you are in the right env, type: `which pip` and you should see it's pointing inside the env folder*

Install numpy, to be done first of all
```
pip install numpy==1.10.4
```

#### Install and config package

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
Edit config.yaml file if needed.

Eventually run
```
stream2segment
```

### Installation Notes:

On Ubuntu 12.10, there might be problems with libxml (`version libxml2_2.9.0' not found`)
Move the file or create a link in the proper folder. The problem has been solved looking at
http://phersung.blogspot.de/2013/06/how-to-compile-libxml2-for-lxml-python.html

On Ubuntu 14.04 
All following issues should be solved  by installing the prerequisites listed above.
However,
- For numpy installation problems (such as `Cannot compile 'Python.h'`) , the fix
has been to update gcc and install python2.7-dev: 
	```sudo apt-get update
	sudo apt-get upgrade gcc
	sudo apt-get install python2.7-dev```
	For details see http://stackoverflow.com/questions/18785063/install-numpy-in-python-virtualenv
- For matplotlib problems, `libpng-dev libfreetype6-dev` are required (see http://stackoverflow.com/questions/25593512/cant-install-matplotlib-using-pip and http://stackoverflow.com/questions/28914202/pip-install-matplotlib-fails-cannot-build-package-freetype-python-setup-py-e)
- For scipy problems, `build-essential gfortran libatlas-base-dev` are required for scipy (see http://stackoverflow.com/questions/2213551/installing-scipy-with-pip/3865521#3865521)
- For lxml problems, `libxml2-dev libxslt-dev` are required (see here: http://lxml.de/installation.html)
