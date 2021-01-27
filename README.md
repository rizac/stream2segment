# <img align="left" height="30" src="https://www.gfz-potsdam.de/fileadmin/gfz/medien_kommunikation/Infothek/Mediathek/Bilder/GFZ/GFZ_Logo/GFZ-Logo_eng_RGB.svg"> Stream2segment <img align="right" height="50" src="https://www.gfz-potsdam.de/fileadmin/gfz/GFZ_Wortmarke_SVG_klein_en_edit.svg">

A Python project to download, process and visualize event-based seismic waveform segments,
specifically created to manage massive amounts of data.

<sub>**Hint**: for massive downloads (as a rule of thumb: >= 1 million segments) we
suggest to use postgres as database, and we **strongly** suggest to run the program on 
computers with at least **16GB** of RAM.</sub>

The key aspects with respect to widely-used similar applications are:

* A database storage (sqlite or postgres) for downloaded data and metadata. Downloading 
  hundreds of thousands, if not millions, of waveform segments on your file system to 
  inspect, select and process them accessing their metadata, is so unfeasible to be in 
  most cases virtually impossible. For this kind of tasks *a database is not polluting your 
  computer with huge, complex and unexplorable directory structures*, stores everything 
  more efficiently, preventing data and metatada inconsistency by design,  allowing a far
  more powerful and efficient data selection and accessibility (see below) and several 
  users to work on the same database remotely, if needed. *Stream2segment uniquely read 
  each downloaded waveform (without performance degradation)* and runs diagnostics (e.g.,
  percentage of downloaded data, gaps/overlaps) which will be saved to the database in order
  to allow a fast and unique selection of suitable segments on both segment's data and 
  metadata.

* A very efficient and straightforward interface for processing downloaded data in any
  kind of custom code (e.g., the processing module described below, Jupyter notebook):
  each segment is exposed as a simple Python object with a list of easily accessible
  attributes denoting data, metadata and related objects (segment's station, channel,
  event, and so on), which also makes the selection of suitable segments to process
  incredibly unique, powerful and easy to use. The selection can be performed on all
  segments attributes, it exploits under the hood the efficiency of the Object-relational
  mapping library to issue SQL `select` commands, but it does not require any specific
  database knowledge thanks to a simplified and custom syntax documented in any generated
  template and in the  [Usage](#usage) section

* An integrated processing environment to get any user-dependent output from the
  downloaded data. Write your own code in a processing module (following few simple
  instructions), run it with the `s2s process` command, and Stream2segment takes care of
  executing the code on all selected segments, interacting with the database for you while
  displaying progress bars, estimated available time, and handling errors. The selection
  of segments to process is implemented via a simple and powerful expressions syntax in a
  config YAML file to be passed to `s2s process`. Few templates (command `s2s init`)
  provide the user with editable examples for two typical scenarios: create tabular file
  outputs (e.g., CSV, HDF), or store on the local file system the processed waveform
  segments.

* A visualization tool to show downloded and optionally customized processed segments in
  a web browser Graphical User Interface (GUI) by means of Python web framework and 
  Javascript libraries (thus, with no external programs required). As for the processing
  module, write your own code (following few simple instructions) run it with the
  `s2s show` command, and Stream2segment takes care of showing the plots *you implemented
  in your code*. The user can also set class labels to make the GUI a hand-labelling tool
  for supervised classification problems, or to simply label special segments for easy 
  election
  
  | The GUI produced with the `show` command  | The dynamic HTML page produced with the `utils dstats` command  |
  | --- | --- |
  | ![](https://geofon.gfz-potsdam.de/software/stream2segment/processgui.png) | ![](https://geofon.gfz-potsdam.de/software/stream2segment/s2s_dinfogui.png)|
  | (image linked from https://geofon.gfz-potsdam.de/software/stream2segment/) | (image linked from https://geofon.gfz-potsdam.de/software/stream2segment/) |


**Citation (Software):**
> Zaccarelli, Riccardo (2018): Stream2segment: a tool to download, process and visualize event-based seismic waveform data. V. 2.7.3. GFZ Data Services.

[http://doi.org/10.5880/GFZ.2.4.2019.002](http://doi.org/10.5880/GFZ.2.4.2019.002)


**Citation (Research article):**
> Riccardo Zaccarelli, Dino Bindi, Angelo Strollo, Javier Quinteros and Fabrice Cotton. Stream2segment: An Open‐Source Tool for Downloading, Processing, and Visualizing Massive Event‐Based Seismic Waveform Datasets. *Seismological Research Letters* (2019)

[https://doi.org/10.1785/0220180314](https://doi.org/10.1785/0220180314)


## Usage

Please refer to the **[github documentation](https://github.com/rizac/stream2segment/wiki)**


## Installation

This program has been installed and tested on Ubuntu (14 and later) and macOS
(El Capitan and later).

In case of installation problems, we suggest you to proceed in this order:

 1. Look at [Installation Notes](#installation-notes) to check if the problem
    has already ben observed and a solution proposed
 2. Google for the solution (as always)
 3. [Ask for help](https://github.com/rizac/stream2segment/issues)


### Requirements (external software)

In this section we assume that you already have Python (**3.5 or later**) 
and the required database software. If you plan to use SQLite, which is generally
used to save all data in a local file, you probably do not need to bother, as
the required package `sqlite3` is generally pre-installed (but check it beforehand).
If you plan to use Postgres, you might use a remote machine (e.g., a server where
Postgres is already installed or where you can delegate its installation),
or otherwise you need to install `postgresql` locally.

<!--
In most packages, the management of required external software is generally left
to the user, as it depends on too many factors. We try to help collecting
in this section and in [Installation Notes](#installation-notes) the feedbacks
from several installations, also be to skip this section and - if you can stand some potential error
during installation  - handle potential errors
due to missing software later, in order to have more control on what you
install or update, and why.
-->

#### macOS

On MacOS (El Capitan and later) all required software is generally already
preinstalled. We suggest you to go to the next step and look at the
[Installation Notes](#installation-notes) in case of problems
(to install software on MacOS, we recommend to use [brew](https://brew.sh/)).

<details>
<summary>Details</summary>

In few cases, on some computers we needed to run one or more of the following
commands (it's up to you to run them now or later, only those really needed):

```
xcode-select --install
brew install openssl
brew install c-blosc
brew install git
```

</details>

#### Ubuntu

Ubuntu does not generally have all required packages pre-installed. The bare minimum
of the necessary packages can be installed with the `apt-get` command:

```
sudo apt-get install git python3-pip python3-dev  # python 3
```

<!-- 
sudo apt-get update

sudo apt-get install git python-pip python2.7-dev  # python 2 -->

<details>
<summary>Details</summary>

In few cases, on some computers we needed to run one or more of the following
commands (it's up to you to run them now or later, only those really needed):

Upgrade `gcc` first:
	
```
sudo apt-get update
sudo apt-get upgrade gcc
```

Then:

```
sudo apt-get update
sudo apt-get install libpng-dev libfreetype6-dev \
	build-essential gfortran libatlas-base-dev libxml2-dev libxslt-dev python-tk
```


</details>

### Cloning repository

Git-clone (basically: download) this repository to a specific folder of your choice:
```
git clone https://github.com/rizac/stream2segment.git
```
and move into package folder:
```
cd stream2segment
```

### Install and activate Python virtualenv

We strongly recommend to use Python virtual environment,
because by isolating all Python packages we are about to install,
we won't create conflicts with already installed packages. 

<!-- * Installation (recommended, but works for Python 3.5+ only) -->

Python (from version 3.3) has a built-in support for virtual environments - venv
(On Ubuntu, you might need to install it first
via `sudo apt-get install python3-venv`).

Make virtual environment in an stream2segment/env directory (env is a convention,
but it's ignored by git commits so better keeping it. You can also use ".env"
which makes it usually hidden in Ubuntu).
```
python3 -m venv ./env
```

<!--
* Installation (all Python versions)

	To install Python virtual environment either use
	[Virtualenvwrapper](http://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation)
	or the more low-level approach `virtualenv`:
	```
	sudo pip install virtualenv
	```
	Make virtual environment in an stream2segment/env directory (env is a convention, but
	it's ignored by git commits so better keep it)
	 ```
	virtualenv env
	 ```
	(on ubuntu 16.04, we got the message 'virtualenv: Command not found.'.
	We just typed: `/usr/local/bin/virtualenv env`)
-->

To activate your virtual environment, type:

 ```
 source env/bin/activate
 ```
or `source env/bin/activate.csh` (depending on your shell)

> <sub>Activation needs to be done __each time__ we will run the program.</sub>
> <sub>To check you are in the right env, type: `which pip` and you should see it's
  pointing inside the env folder</sub>


<details>
	<summary>Installation and activation with Anaconda (click to expand)</summary>

**disclaimer: the lines below might be outdated.
Please refer to the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for details**

Create a virtual environment for your project

  - In the terminal client enter the following where yourenvname (like « env ») is the
    name you want to call your environment, and replace x.x with the Python version you
    wish to use. (To see a list of available Python versions first, type conda search
    "^python$" and press enter.)
	```
	Conda create –n yourenvname python=x.x anaconda
	```
  - Press « y » to proceed

Activate your virtual environment

  - ```$source activate env```
  - To deactivate this environment, use ```$source deactivate```

</details>

### Install Stream2segment Python package

**Important reminders before installing**: 
  - From now on you are supposed to be in the stream2segment directory,
     (where you cloned the repository) with your Python virtualenv activated
  - In case of errors, check the [Installation notes below](#installation-Notes)

Installation first installs all *requirements* (i.e., required external Python
packages) and then this package, and can be performed in a single command in two ways:

1. If you already have other Python packages installed in the virtual environment, run:
   ```
   pip install -e .
   ```
   (the -e is optional, [it makes the package editable](https://pip.pypa.io/en/stable/reference/pip_install/#install-editable)). 
   This does not re-install already installed requirements if they satisfy Stream2segment
   minimum versions, and it's therefore generally safer in order to avoid conflicts with
   existing packages. However, in all other cases requirements are installed with their
   newest version. Therefore, you might have problems with Stream2segment, if some
   requirements is used with a new untested version. Obviously, we do our best to keep
   everything updated regularly and avoid this case as much as we can.

2. If you plan to only use Stream2segment in an empty virtual environment, run:
   ```
   ./installme
   ```
   (or `./installme-dev` if you want to contribute and/or run tests to check if the program
    will likely work in your system). This installs all requirements with specific version
    "freezed" after successfully running tests, and it's therefore generally safer in order
    to avoid problems with Stream2segment.

<details>

<summary>scripts details</summary>

The scripts above are a shorthand for the following commands, which you can always run by
yourself to have more control on what is happening:

1. Install numpy first (this is an obspy requirement): open the file `requirements.txt`
   and search for the line starting with "numpy". Copy it and pip-install numpy. For
   instance, supposing that the line is `numpy==1.15.4`, then execute on the terminal:
   `pip install numpy==1.15.4`

2. Install via requirements file:
    ```
    pip install -r ./requirements.txt
    ```
    Alternatively, if you want to run tests (recommended to check that everything works
    on your system):
    ```
    pip install -r ./requirements.dev.txt
    ```

3. Install the current package
   ```
   pip install -e .
   ```

   (The `-e` options installs this package as editable, meaning that after making a
   change - e.g. a `git pull` to fetch a new version - you don't need to reinstall it but
   the new version will be already available for use)

</details>

##### Install Jupyter (optional)

If you wish to use the program within Jupyter notebooks, jupyter is not included
in the dependencies. Thus
```
pip install jupyter==1.0.0
```

**The program is now installed. To double check the program functionalities,
we suggest to run tests (see below) and report the problem in case of failure.
In any case, before reporting a problem remember to check first the
[Installation Notes](#installation-notes)**

### Runt tests

Stream2segment has been highly tested (current test coverage is above 90%)
on Python version >= 3.5+ ~~and 2.7~~ (as of 2020, support for Python2 is 
discontinued). Although automatic continuous integration (CI) systems are not
in place, we do our best to regularly tests under new Python versions, when
available. Remember that tests are time consuming (some minutes currently).
Here some examples depending on your needs:

```
pytest -xvvv -W ignore ./tests/
```

```
pytest -xvvv -W ignore --cov=./stream2segment --cov-report=html ./tests/
```

```
pytest -xvvv -W ignore --dburl postgresql://<user>:<password>@localhost/<dbname> --cov=./stream2segment --cov-report=html ./tests/
```

Where the options denote:

- `-x`: stop at first error
- `-vvv`: increase verbosity,
- `-W ignore`: do not print Python warnings issued during tests. You can omit the `-W`
  option to turn warnings on and inspect them, but consider that a lot of redundant
  messages will be printed: in case of test failure, it is hard to spot the relevant error
  message. Alternatively, try `-W once` - warn once per process - and `-W module` -warn
  once per calling module.
- `--cov`: track code coverage, to know how much code has been executed during tests, and
  `--cov-report`: type of report (if html, you will have to opend 'index.html' in the
  project directory 'htmlcov')
- `--dburl`: Additional database to use.
  The default database is an in-memory sqlite database (e.g., no file will be created),
  thus this option is basically for testing the program also on postgres. In the example,
  the postgres is installed locally (`localhost`) but it does not need to.
  *Remember that a database with name `<dbname>` must be created first in postgres, and
  that the data in any given postgres database will be overwritten if not empty*


## Installation Notes:

- (update January 2021) On MacOS (version 11.1, with Python 3.8 and 3.9):

  - if the installation fails with a lot of printout and you spot a
    "Failed building wheel for psycopg2", try to execute:
    ```
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/opt/openssl/lib/ && pip ./installme-dev
    ```
    (you might need to change the path of openssl below). Credits
    [here](https://stackoverflow.com/a/61159643/3526777) and
    [here](https://stackoverflow.com/a/39800677/3526777))
 
  - If the error message is "Failed building wheel for tables",
    then `brew install c-blosc` and re-run `installme-dev` installation command
    (with the `export` command above, if needed)
 

- If you see (we experienced this while running tests, thus we can guess you should see
  it whenever accessing the program for the first time):
  ```
  This system supports the C.UTF-8 locale which is recommended.
  You might be able to resolve your issue by exporting the
  following environment variables:

    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
  ```
  Then edit your `~/.profile` (or `~/.bash_profile` on Mac) and put the two lines starting
  with 'export', and execute `source ~/.profile` (`source ~/.bash_profile` on Mac) and
  re-execute the program.  

- On Ubuntu 12.10, there might be problems with libxml (`version libxml2_2.9.0' not found`). 
  Move the file or create a link in the proper folder. The problem has been solved looking
  at http://phersung.blogspot.de/2013/06/how-to-compile-libxml2-for-lxml-python.html

All following issues should be solved by installing all dependencies as described in
the section [Prerequisites](#prerequisites). If you did not install them, here the solutions
to common problems you might have and that we collected from several Ubuntu installations:

- For numpy installation problems (such as `Cannot compile 'Python.h'`) , the fix 
  has been to update gcc and install python3-dev (python2.7-dev if you are using Python2.7,
  discouraged): 
  ```
  sudo apt-get update
  sudo apt-get upgrade gcc
  sudo apt-get install python3-dev
  ```
   For details see [here](http://stackoverflow.com/questions/18785063/install-numpy-in-python-virtualenv)
 
 - For scipy problems, `build-essential gfortran libatlas-base-dev` are required for scipy.
   For details see [here](http://stackoverflow.com/questions/2213551/installing-scipy-with-pip/3865521#3865521)
 
 - For lxml problems, `libxml2-dev libxslt-dev` are required. For details see [here](http://lxml.de/installation.html)
 
 - For matplotlib problems (matplotlib is not used by the program but from imported libraries),
   `libpng-dev libfreetype6-dev` are required. For details see
   [here](http://stackoverflow.com/questions/25593512/cant-install-matplotlib-using-pip) and
   [here]( http://stackoverflow.com/questions/28914202/pip-install-matplotlib-fails-cannot-build-package-freetype-python-setup-py-e)

## Developer(s) notes:

- Although PEP8 recommends 79 character length, the program used initially a 100
  characters max line width (we are now moving back to 79 character length.
  It seems that [among new features planned for Python 4 there is
  an increment to 89.5 characters](https://charlesleifer.com/blog/new-features-planned-for-python-4-0/).
  If true, we might stick to that in the future) 
  
- The program can be also installed via the usual way by ignoring `requirements.txt` and
  just typing:
  ```
  pip install -e .
  ```
  The difference is that this way the newest version of all dependencies are downloaded,
  whereas with `requirements.txt` (see above) all dependencies are installed with the
  specific version on which tests passed successfully. Therefore, the latter approach is
  a lot safer. From times to times, it is nevertheless necessary to update the dependencies,
  which would also make `pip install` more likely to work (at least for some time). The
  procedure is:
  ```
	pip install -e .
	pip freeze > ./requirements.tmp
	pip install -e .[dev,test]
	pip freeze > ./requirements.dev.tmp
  ```
  Run tests (see above) with warnings on: fix what might go wrong, and eventually you can
  replace the old `requirements.txt` and `requirements.dev.txt` with the `.tmp` file
  created. 

<!--
## Misc:

### sqlitebrowser
The program saves data on a sql database (tested with postresql and sqlite). If sqlite is
used as database, to visualize the sqlite content you can download sqlitebrowser
(http://sqlitebrowser.org/). The installation on Mac is straightforward (use brew cask or
go to the link above) whereas on Ubuntu can be done as follows:
```
sudo add-apt-repository ppa:linuxgndu/sqlitebrowser
sudo apt-get install sqlitebrowser
```

### matplotlibrc

A `matplotlibrc` file is included in the main root package. As said, matplotlib is not
used by the program but from imported libraries, The included file sets the backend to
'Agg' so that we hide the "Turning interactive mode on" message (for Mac users)
-->
