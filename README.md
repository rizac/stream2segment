# <img align="left" height="30" src="https://www.gfz-potsdam.de/fileadmin/gfz/medien_kommunikation/Infothek/Mediathek/Bilder/GFZ/GFZ_Logo/GFZ-Logo_eng_RGB.svg"> Stream2segment <img align="right" height="50" src="https://www.gfz-potsdam.de/fileadmin/gfz/GFZ_Wortmarke_SVG_klein_en_edit.svg">

A Python project to download, process and visualize event-based seismic waveform 
segments, specifically designed to manage big volumes of data.

The key aspects with respect to widely-used similar applications is the use of
a Relational database management system (RDBMS) to store downloaded data and 
metadata. **Note** the program supports SQLite and Postgres RDBMS. For massive 
  downloads (as a rule of thumb: &ge; 1 million segments) we suggest to 
  use Postgres, and we **strongly** suggest to run the program on computers with at 
  least **16GB** of RAM.
  
The main advantages of this approach are: 

* **Storage efficiency**: no huge amount of files, no complex, virtually 
  unusable directory structures. Moreover, a database prevents data and metatada 
  inconsistency by design, and allows more easily to track what has already 
  been downloaded in order to customize and improve further downloads

* **Simple Python objects representing stored data and relationships**, easy 
  to work with in any kind of custom code. For instance, a segment is 
  represented by a `Segment` object with its data, metadata and related objects 
  easily accessible through its attributes, e.g., `segment.stream()`, 
  `segment.maxgap_numsamples`, `segment.event.magnitude`, 
  `segment.station.network`, `segment.channel.orientation_code` and so on.
  
* **A powerful segments selection** made even easier by means of a simplified
  syntax: map any attribute described above to a selection expression
  (e.g. `segment.event.magnitude: "[4, 5)"`) and with few lines you can compose 
  complex database queries such as e.g., "get all downloaded segments within a 
  given magnitude range, with no gaps, enough data (related to the requested 
  data), from broadband channels only and a given specific network"

Each download is highly customizable with several parameters for any step required
(event, data center, station and waveform data download). In addition, as data 
is never downloaded per se, Stream2segment helps the whole workflow with:

* **An integrated processing environment** to get any user-dependent output (e.g., 
  tabular output such as CSV or HDF files). Write your segment selection 
  in a configuration (YAML) file, and your own code in a processing (Python) 
  module.
  Pass both files to the `s2s process` command, and Stream2segment takes care of
  executing the code on all selected segments, interacting with the database for 
  you while displaying progress bars, estimated available time, and handling 
  errors.

* **A visualization tool** to show *any* kind of user defined plot from each selected
  segment. Similar to the processing case above, write your selection in 
  the configuration file and the code of your own plots in the processing (or 
  any Python) module. 
  Pass both files to the `s2s show` command, and Stream2segment takes care of 
  visualizing your plots in a *web browser* Graphical User Interface (GUI) with
  no external programs required. The GUI can also be used to label segments with 
  user-defined classes in order to refine the segments selection later, or for 
  creating datasets in machine-learning supervised classification problems

* **Several utilities** to interact with the database, print download reports
  or show interactive stats on web GUI maps (see command `s2s utils`)
  
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

A detailed documentation is available online in the
**[github wiki page](https://github.com/rizac/stream2segment/wiki)**, but
you can also simply start the program via the command `init` ( 
`s2s init --help` for details) which creates several examples files to run
the program. These files contain roughly the same online documentation  
in form of comments to code and parameters that you can immediately start to
configure and modify. These files are: 

 - A download configuration file (in YAML syntax with all parameters documented)
   to start the download routine:
   ```bash
   s2s download -c <config_file> ...
   ```
 - Two modules (Python files) with relative configurations (YAML files)
   to be passed to the processing routine:
   ```bash
   s2s process -c <config_file> -p <processing_module> ...
   ``` 
   or the visualization routine:
   ```bash
   s2s show -c <config_file> -p <processing_module> ...
   ``` 
 - A Jupyter notebook tutorial with examples, for user who prefer 
   this approach when working with downloaded data

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
and the required database software. The latter should not needed if you use
[SQLite](https://docs.python.org/3/library/sqlite3.html) or if the
database is already installed remotely, so basically you are concerned only if you
need to download data locally (on your computer) on a Postgres database.

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

To install the package, you should run as usual `pip install`, but because
some required packages must unfortunately be installed in a specific order,
we implemented a script that handles that and can be invoked exactly as `pip install`:
 
```
./pipinstall <options...> .
```

<details>
<summary>script details (if you want to run each
`pip install` command separately to have more control)</summary>

`pipinstall` is simply a shorthand for several `pip install` commands, run in these
specific order:

1. Install pre-requisites as `pip install --upgrade pip setuptools wheel`

1. Install numpy first (this is an obspy requirement): either `pip install numpy` or,
   if you want to use a requirements file, extracting (e.g. via `grep`) the specific
   numpy version in the file (e.g. `numpy==1.15.4`), and then executing
   `pip install numpy==1.15.4`

2. Running `pip install` with *exactly the same arguments* provided to the script. E.g.
   `pipinstall -e .` executes `pip install -e .` 

</details>

The `pipinstall` argument `.` means "install this directory" (i.e., stream2segment) and
can be enhanced with extra packages. For instance, if you want to install Jupyter in order 
to work with Stream2segment downloaded data in a notebook, then type:
```
    ./pipinstall <options...> ".[jupyter]"
```
If you want to install additional packages needed for testing (install in dev mode)
and be able to push code and/or run tests, then type:
```
    ./pipinstall <options...> ".[dev]"
```
(You can also provide both: `".[dev,jupyter]"`. Quotes were necessary on some
specific macOS with `zsh`, in other OSs or shell languages might not be needed)

The `<options...>` are the usual `pip install` options. The two more important are usually:

 `-e` This [makes the package editable](https://pip.pypa.io/en/stable/reference/pip_install/#install-editable).
  A typical advantage of an editable package is that when you run `git pull` to fetch a new
  version that does not need new requirements (e.g. a bugfix), you don't need to reinstall
  it but the new version will be already available for use
 
 `-r ./requirements.txt`: install requirements with specific versions. `pip install` by
  default skips already installed requirements if they satisfy Stream2segment minimum
  versions. With the `-r` option instead, requirements are installed with "tested" versions
  (i.e., those "freezed" after successfully running tests), which should generally be safer
  for obvious reasons. However, some versions in the requirements might not be (yet?) supported
  in your computer, some might be in conflicts with the requirements of other packages you
  installed in the virtualenv, if any. You can try this option and then remove it, in case
  of problems. In any case, do not use this option if you plan to install other
  stuff alongside stream2segemtn on the virtualenv. 
  
  (There is also a `./requirements.dev.txt`
  that installs also the dev-related packages, similar to `".[dev]"`, but with specific
  exact versions.

  
<!--
Installation first installs all *requirements* (i.e., required external Python
packages) and then this package, and can be performed in a single command in two ways:

1. If you already have other Python packages installed in the virtual environment, run:
   ```
   pip install -e .
   ```
   (the -e is optional, [it makes the package editable](https://pip.pypa.io/en/stable/reference/pip_install/#install-editable)). 
   This does not reinstall already installed requirements if they satisfy Stream2segment
   minimum versions, and it's therefore generally safer in order to avoid conflicts with
   existing packages. However, in all other cases requirements are installed with their
   newest version. Therefore, you might have problems with Stream2segment, if some
   new untested requirement is used (we do our best to keep
   everything updated regularly and avoid this case as much as we can).

2. If you plan to only use Stream2segment in an empty virtual environment, run:
   ```
   ./installme
   ```
   (or `./installme-dev` if you want to contribute and/or run tests to check if the program
    will likely work in your system). This installs all requirements with specific version
    "freezed" after successfully running tests, and it's therefore generally safer in order
    to avoid problems with Stream2segment.


##### Install Jupyter (optional)

If you wish to use the program within Jupyter notebooks, jupyter is not included
in the dependencies. Thus
```
pip install jupyter>=1.0.0
```
-->

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
pytest -xvvv -W ignore --dburl postgresql://<user>:<password>@localhost/<dbname> ./tests/
```

<!--
```
pytest -xvvv -W ignore --cov=./stream2segment --cov-report=html ./tests/
```
-->

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
  characters max line width, which is being reverted to 79 (you might see mixed
  lengths in the modules). It seems that [among new features planned for Python 4 there is
  an increment to 89.5 characters](https://charlesleifer.com/blog/new-features-planned-for-python-4-0/).
  If true, we might stick to that in the future
  
- In the absence of Continuous Integration in place, from times to times, it is necessary
  to update the dependencies, to make `pip install` more likely to work (at least for
  some time). The procedure is:
  ```
	pip install -e .
	pip freeze > ./requirements.tmp
	pip install -e ".[dev]"
	pip freeze > ./requirements.dev.tmp
  ```
  (you could also do it with jupyter). Remember to comment the line of stream2segment
  from each requirements (as it should be installed as argument of pip:
  `pip install <options> .`, and not inside the requirements file).
  Run tests (see above) with warnings on: fix what might go wrong, and eventually you can
  replace the old `requirements.txt` and `requirements.dev.txt` with the `.tmp` file
  created. 

- Updating wiki:
  
  Requirements:
   - 'jupyter installed'.
   - A clone of the repo 'stream2segment.wiki' on the same parent directory of the
     stream2segment repo.
  
  Then:
   - Edit the notebooks in stream2segment/resources/templates:
   - Some new notebook? First choose a meaningful file name, as it will be
     the title of the wiki page (hyphens will be replaced with spaces in the titles).
     Then check that the notebook is tested, i.e. run in "tests/misc/test_notebook"
   - Create .md (markdown) versions of the notebook for the wiki. From the
     stream2segment repository as `cwd` (`F` is the filename without the 'ipnyb' extension):
      ```bash
      F='Using-Stream2segment-in-your-Python-code';jupyter nbconvert --to markdown ./stream2segment/resources/templates/$F.ipynb --output-dir ../stream2segment.wiki 
      ```
      (repeat for every notebook file, e.g. `Using-Stream2segment-in-your-Python-code`)
   - `cd ../stream2segment.wiki`: 
      - Some new notebook? add it in the TOC of 'Home.md' if needed, and `git add`
      - Eventually, `git commit`, `push` as usual. Check online. Done
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
