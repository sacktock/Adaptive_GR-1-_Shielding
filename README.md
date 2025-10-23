# Adaptive GR(1) Shielding

Codebase for the conference paper: "Adaptive GR(1) Specification Repair for Liveness-Preserving Shielding in Reinforcement Learning".

## Reproducing Experiments

Firstly, please follow the installation instructions below. After correctly setting up your environment, all the experiments from the paper can be reproduced by running,
```
python run_all_minepump.py
```
and
```
python run_all_seaquest.py
```
Individual runs can be lauched via the python scripts: ```run_minepump.py``` or ```run_seaquest.py```. Differnt shield implementations can be run via the command line arguments, for exmaple,
```
python run_minepump.py --shield adaptive
```
runs the minepump experiment with our adaptive shield framework.

## Logging

We use [TensorBoard](https://www.tensorflow.org/tensorboard) for logging metrics. tensorBoard logging can be turned on via,
```
python run_minepump.py --tensorboard --logdir "./logdir/minepump/adaptive_shield" --shield adaptive 
```
where ```--logdir``` specifies teh location that the tensorboard logs are saved to. The results can be viewed by launching TensorBoard via the command line,
```
tensorboard --logdir "./logdir/minepump/adaptive_shield"
```
and navigating to (http://localhost:6006/). 

## Installation Instructions

First we recommend you setup a virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Creating a new environment from the ```environment.yml``` file,

```
conda env create -f environment.yml
```

### Spectra CLI

Spectra CLI is a crucial dependency, pelase install it via the [installation instructions](https://github.com/SpectraSynthesizer/spectra-cli). This requires building from source with your corresponding Java distribution. Alternatively, please feel free to use our binaries, provided in ```bin```, these have been built with Java 23, to run these binaries you will need JDK/JRE 23.

#### Ubuntu (Java 23):
At the time of writing, the following won't work out of the box for Ubuntu 22.04,
```
sudo apt install openjdk-23-jdk -y
```
Thus we need to install it via Temurin,
```
sudo apt-get update
sudo apt-get install -y wget apt-transport-https gnupg

wget -O- https://packages.adoptium.net/artifactory/api/gpg/key/public \
 | sudo gpg --dearmor -o /usr/share/keyrings/adoptium-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/adoptium-archive-keyring.gpg] \
https://packages.adoptium.net/artifactory/deb $(. /etc/os-release; echo $VERSION_CODENAME) main" \
 | sudo tee /etc/apt/sources.list.d/adoptium.list

sudo apt-get update
sudo apt-get install -y temurin-23-jdk
```
Make Java 23 the default
```
sudo update-alternatives --config java
sudo update-alternatives --config javac
```

#### What to do with the binaries

Please place all the binaries in an appropriate place, e.g.,

```
mv spectra-cli.jar ~/SpectraTools/spectra-cli.jar
mv spectra_unrealizable_cores.jar ~/SpectraTools/spectra_unrealizable_cores.jar
mv spectra_all_unrealisable_cores.jar ~Spectra/Tools/spectra_all_unrealisable_cores.jar
mv spectra_toolbox.jar ~/SpectraTools/spectra_toolbox.jar
mv spectra-executor.jar ~/SpectraTools/spectra-executor.jar
```

### Install clingo version 5.8.0

Version 5.8.0 of Potassco's Clingo tool should be installed. As mentioned on its [installation page](https://potassco.org/clingo/), you can install it via conda or pip. In the command line run,
```
> clingo -v
```
you should see the following message,
```
clingo version 5.8.0 (6d1efb6)
Address model: 64-bit

libclingo version 5.8.0
Configuration: with Python 3.10.12, with Lua 5.3.6

libclasp version 3.4.0 (libpotassco version 1.2.0)
Configuration: WITH_THREADS=1
Copyright (C) Benjamin Kaufmann

License: The MIT License <https://opensource.org/licenses/MIT>
```
We recommend you install clingo via your machine; since Lua is required for our usage, conda likely won't work and you may have to build from source.

#### Ubuntu:
Using apt
```bash
sudo add-apt-repository ppa:potassco/stable
sudo apt update
sudo apt install clingo
```

#### MacOS:
Using Homebrew
```bash
brew install clingo
```

### ILASP and FastLAS

**Install ILASP version 4.4.0:** Download the appropriate version of the ILASP learner from their
[releases page](https://github.com/ilaspltd/ILASP-releases/releases).

**Install FastLAS version 2.1.0:** Download the appropriate version of the FastLAS learner from their
[releases page](https://github.com/spike-imperial/FastLAS/releases).

Alternatively, we provide binaries for ILASP and FastLAS in ```bin```, place both binaries in an appropriate place, e.g., 
```
mv ./ILASP ~/bin
mv ./FastLAS ~/bin
```

### Install Spot

Spot is a tool that is required to process ltl as strings. To install it, follow its installation instructions on its [installation page](https://spot.lre.epita.fr/install.html)

For simplicity we recommend you install with conda in your virtual environment,
```
conda create --name gr1_shield python=3.10 # adjust as desired
conda activate gr1_shield
conda install -c conda-forge spot
```

### Additional python dependencies

Additional python dependencies can be install via pip or conda, 
```
conda env create -f environment.yml
```
or updating an existing environment,
```
conda env update -f environment.yml --prune
```
or with pip,
```
pip install -r requirements.txt
```

### Custom python dependencies

Our custom python library ```py-ltl```, is not supported by conda and may not be installed following the previous instructions, please install it with,
```
pip install -i https://test.pypi.org/simple/ py-ltl
```

### Setting up shield/config.py

The final step is to make sure the paths in ```shield/config.py``` are correct for your system setup, make sure ```PATH_TO_JVM``` correctly references the ```libjvm.so``` file, e.g., for ubuntu,
```
PATH_TO_JVM = "/usr/lib/jvm/temurin-23-jdk-amd64/lib/server/libjvm.so"
``` 
Similarly, make sure ```PROJECT_PATH``` is set to the current directory and ```LOG_FOLDER``` is set to an appropriate path.

### Testing your setup

To test the adaptive shield repair, please run,
```
python -m shield.adaptive_controller_shield
```
you should not see any error output if the installation was successful.

# Reproducing our results

You can directly reproduce our results by running either,
```
python run_all_minepump.py
```
or,
```
python run_all_seaquest.py
```