![logo](./figures/DreamDIA.jpg)

**Software for data-independent acquisition (DIA) data analysis with deep representation features and FDR-controlled match-between runs (DreamDIAlignR).**



### New Feature of DreamDIA3: DreamDIAlignR

DreamDIAlignR is a novel cross-run peptide-centric analysis workflow that allows for consistent cross-run peak picking and FDR-controlled peak scoring.



### 1. Docker image & quick example

*We recommend using Docker containers to simplify installation and avoid complex compilation processes.*

1. Pull the docker image from DockerHub.

```shell
docker pull mingxuangao/dreamdia:v3.2.0
```

2. Start a container.

```shell
docker run -it --name dreamdia_example --gpus all -v /YOUR/OWN/WORK/PATH:/tmp/work mingxuangao/dreamdia:v3.2.0 /bin/bash
```

* The `--gpus all` argument enables GPU support within the container, which is essential for running the latest version of DreamDIA.

3. Activate the conda environment for DreamDIA.

```shell
conda activate dreamdia
```

4. (optional) Test the availability of GPUs.

```shell
python /root/check_gpus.py
```

5. Run the demo.

```shell
# Enter the working directory
cd

# Run DreamDIA peptide peak identification module
python DreamDIA/DreamDIA.py dreamscore --file_dir example_data/raw_data --lib example_data/Spyogenes_library.tsv --out dreamscore_out

# Run DreamDIAlignR
python DreamDIA/DreamDIA.py dreamprophet --dream_dir dreamscore_out --out dreamdia_out --dreamdialignr --r_home /root/miniconda3/envs/dreamdia/bin/R 
```

### 2. Build from source

##### Requirements

```
Linux
python >= 3.6.0
pyteomics
numpy
pandas
seaborn
cython
scikit-learn
tensorflow
keras-gpu >= 2.4.3
statsmodels
xgboost
networkx
rpy2 >= 3.5
R >= 4.2.0
```

If `.raw` files are to be directly input into DreamDIA on Linux systems, [mono](https://www.mono-project.com/download/stable/#download-lin) must be installed.

We recommend using [Anaconda](https://www.anaconda.com/products/individual#Downloads) to set up the environment and install the necessary libraries as outlined below.

```shell
# Initiate a conda virtual environment called "dreamdia"
conda create -n dreamdia python=3.6.12

# Activate the "dreamdia" virtual environment
conda activate dreamdia

# Install the libraries
conda install -y keras-gpu
conda install -y scikit-learn
conda install -y py-xgboost-cpu
conda install -y cython
conda install -y seaborn
conda install -y statsmodels
conda install -y pyteomics -c bioconda
conda install -y networkx
conda install -y r-base=4.2.1 -c conda-forge
conda install -y rpy2
```

##### Download

https://github.com/xmuyulab/DreamDIA/releases/tag/v3.2.0

##### Installation

```shell
cd DreamDIA
bash build.sh
```

### 3. Quick start

The latest version of DreamDIA analysis workflow consists of two steps: `dreamscore` and `dreamprophet`.

```shell
# Step1
python DreamDIA-vXXX/DreamDIA.py dreamscore --help

# Step2
python DreamDIA-vXXX/DreamDIA.py dreamprophet --help
```

`dreamscore` identifies peptide peaks for each provided run ,while `dreamprophet` needs the output of the first step and performs optional match-between-runs and statistical analysis.

`dreamprophet` does not perform match-between-runs by default. However, if the `--dreamdialignr` and `--r_home` options are specified, the DreamDIAlignR algorithm will be activated and executed.

### 4. Notes

#### 1. DIA raw data files

* Centroided .mzML or .mzXML files are supported at any time. 
* If .raw files are going to be fed directly to DreamDIA on Linux systems, [mono](https://www.mono-project.com/download/stable/#download-lin) must be installed first for the data format conversion by ThermoRawFileParser.

All raw data files should be in one folder as shown below. 

```
# rawdata_dir/
	rawdata_1.mzML
	rawdata_2.mzXML
	rawdata_3.raw
```

#### 2. Spectral libraries

Only .tsv libraries are supported. All of the columns required by DreamDIA are listed in `DreamDIA/lib_col_settings`. Users can modify this file to adjust their own spectral libraries.

#### 3. output

DreamDIA outputs peptide and protein identification and quantification results. An empty directory is suggested for the `--out` argument to save all of the output files.

#### 4*. Advanced: train your own deep representation models

See the guidance in `Train_customized_models.ipynb` to train your own deep representation models.

### 5. Cite this article

[1] Gao, M., Yang, W., Li, C. et al., Deep representation features from DreamDIA<sup>XMBD</sup> improve the analysis of data-independent acquisition proteomics. *Commun Biol* 4, 1190 (2021). https://doi.org/10.1038/s42003-021-02726-6

[2] Gao, M., Gupta, S., Yang, W. et al., Scoring information integration with statistical quality control enhanced cross-run analysis of data-independent acquisition proteomics data. *BioRxiv*, 2024. https://www.biorxiv.org/content/10.1101/2024.12.19.629475v1

