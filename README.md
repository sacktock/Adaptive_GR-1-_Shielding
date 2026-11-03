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

Individual runs can be launched via the python scripts: `run_minepump.py` or `run_seaquest.py`. Different shield implementations can be run via the command line arguments, for example,

```
python run_minepump.py --shield adaptive
```

runs the minepump experiment with our adaptive shield framework.

## Logging

We use [TensorBoard](https://www.tensorflow.org/tensorboard) for logging metrics. TensorBoard logging can be turned on via,

```
python run_minepump.py --tensorboard --logdir "./logdir/minepump/adaptive_shield" --shield adaptive
```

where `--logdir` specifies the location that the tensorboard logs are saved to. The results can be viewed by launching TensorBoard via the command line,

```
tensorboard --logdir "./logdir/minepump/adaptive_shield"
```

and navigating to (http://localhost:6006/).

## Installation Instructions

Create a conda environment, install Spot into it, then install the Python packages with pip. Experiments are run with `python` on the activated environment's `PATH`.

### Conda environment

Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), then from this repository:

```
conda env create -f environment.yml
conda activate adaptive-gr1
```

`environment.yml` only selects Python 3.12 and pip. The environment name is `adaptive-gr1`.

### Spot

Spot processes LTL formulas and provides `ltlfilt`. Install it into the activated environment. See the [Spot install page](https://spot.lre.epita.fr/install.html) for other options.

```
conda install -c conda-forge spot
```

### Python dependencies

```
pip install -r requirements.txt
```

`requirements.txt` is a set of minimum versions, plus an exact pin of MASA-Safe-RL 0.3.13 (`Requires-Python: >=3.12`). `py-ltl` is included from TestPyPI by that file, so it does not need a separate install. The pip command installs the CPU build of PyTorch. A CUDA build of JAX, if you want one, is a later optional install and should not use `-U`.

### Java and Spectra

Spectra is required to synthesise shields. Build the jars from [spectra-cli](https://github.com/SpectraSynthesizer/spectra-cli) with JDK 23, or use jars you have already built with Java 23. This repository does not ship those binaries.

The Python code loads three jars:

- `spectra-cli.jar`
- `spectra_toolbox.jar`
- `spectra-executor.jar`

A matching layout, which is also the default when the environment variables below are unset:

```
mv spectra-cli.jar ~/SpectraTools/spectra-cli.jar
mv spectra_toolbox.jar ~/SpectraTools/spectra_toolbox.jar
mv spectra-executor.jar ~/SpectraTools/spectra-executor.jar
```

#### Ubuntu (Java 23)

At the time of writing, the following won't work out of the box for Ubuntu 22.04,

```
sudo apt install openjdk-23-jdk -y
```

Install Temurin instead:

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

Make Java 23 the default:

```
sudo update-alternatives --config java
sudo update-alternatives --config javac
```

### Clingo 5.8.0

Install Clingo 5.8.0 from [Potassco](https://potassco.org/clingo/). Lua is required, and the conda build typically does not include it, so install Clingo on the machine (apt, Homebrew, or a source build) rather than into the conda environment.

Check with:

```
clingo -v
```

You should see Clingo 5.8.0, including a Lua version in the configuration line.

#### Ubuntu

```bash
sudo add-apt-repository ppa:potassco/stable
sudo apt update
sudo apt install clingo
```

#### macOS

```bash
brew install clingo
```

### ILASP and FastLAS

**ILASP 4.4.0:** download a binary from the [ILASP releases](https://github.com/ilaspltd/ILASP-releases/releases).

**FastLAS 2.1.0:** download a binary from the [FastLAS releases](https://github.com/spike-imperial/FastLAS/releases).

The defaults expect both on `~/bin`:

```
mv ILASP ~/bin
mv FastLAS ~/bin
```

### Environment variables

`spec_repair/config.py` reads the paths below from the environment when Python starts. Set them before you run a script. An unset variable falls back to the default in that file.

**Set these to match your machine.** The built-in defaults are one specific checkout and one Ubuntu JDK, so a new setup needs both:

- `GR1_PROJECT_PATH` — this repository
- `GR1_JVM_PATH` — the JVM shared library used by JPype. On Linux that is `libjvm.so` (often `.../lib/server/libjvm.so`). On macOS it is `libjvm.dylib`. On Windows it is `jvm.dll` (often `...\bin\server\jvm.dll`).

**Set these when the tools are not in the default locations** listed in the table. If you used `~/SpectraTools` and `~/bin` as above, leave them unset.

- `GR1_SPECTRA_CLI`
- `GR1_SPECTRA_TOOLBOX`
- `GR1_SPECTRA_EXECUTOR`
- `GR1_ILASP`
- `GR1_FASTLAS`

**Defaults** when the variable is unset:

| Variable | Default |
|---|---|
| `GR1_PROJECT_PATH` | `/mnt/c/Users/alexw/OneDrive/Imperial/Projects/Adaptive-GR(1)-Shielding` |
| `GR1_JVM_PATH` | `/usr/lib/jvm/temurin-23-jdk-amd64/lib/server/libjvm.so` |
| `GR1_SPECTRA_CLI` | `~/SpectraTools/spectra-cli.jar` |
| `GR1_SPECTRA_TOOLBOX` | `~/SpectraTools/spectra_toolbox.jar` |
| `GR1_SPECTRA_EXECUTOR` | `~/SpectraTools/spectra-executor.jar` |
| `GR1_ILASP` | `~/bin/ILASP` |
| `GR1_FASTLAS` | `~/bin/FastLAS` |

`clingo`, `java`, and `ltlfilt` are taken from `PATH`. They are not environment variables.

#### Conda-scoped

These values are stored on the `adaptive-gr1` environment and exported whenever it is activated. Reactivate after changing them. Replace the two highlighted paths; add the Spectra and learner variables only when you are not using the defaults.

```bash
conda activate adaptive-gr1
conda env config vars set \
  GR1_PROJECT_PATH="/path/to/Adaptive-GR(1)-Shielding" \
  GR1_JVM_PATH="/usr/lib/jvm/temurin-23-jdk-amd64/lib/server/libjvm.so"
conda deactivate
conda activate adaptive-gr1
conda env config vars list
```

Example that also overrides the tool locations:

```bash
conda env config vars set \
  GR1_SPECTRA_CLI="$HOME/SpectraTools/spectra-cli.jar" \
  GR1_SPECTRA_TOOLBOX="$HOME/SpectraTools/spectra_toolbox.jar" \
  GR1_SPECTRA_EXECUTOR="$HOME/SpectraTools/spectra-executor.jar" \
  GR1_ILASP="$HOME/bin/ILASP" \
  GR1_FASTLAS="$HOME/bin/FastLAS"
```

#### Global

Bash, in `~/.bashrc`:

```bash
export GR1_PROJECT_PATH="/path/to/Adaptive-GR(1)-Shielding"
export GR1_JVM_PATH="/usr/lib/jvm/temurin-23-jdk-amd64/lib/server/libjvm.so"
```

PowerShell, for the current user (new terminals pick this up):

```powershell
[System.Environment]::SetEnvironmentVariable("GR1_PROJECT_PATH", "C:\path\to\Adaptive-GR(1)-Shielding", "User")
[System.Environment]::SetEnvironmentVariable("GR1_JVM_PATH", "C:\Program Files\Eclipse Adoptium\jdk-23\bin\server\jvm.dll", "User")
```

Open a new terminal after either change. A conda-scoped value overrides a global one while `adaptive-gr1` is active, because conda exports it on activate.

### Testing your setup

From the repository root, with `adaptive-gr1` activated:

```
python -m shield.adaptive_controller_shield
```

You should not see any error output if the installation was successful.
