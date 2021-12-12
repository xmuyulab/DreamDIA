![logo](./figures/Dream-DIA.jpg)

Software for data-independent acquisition (DIA) data analysis with deep representation features.

## 1. Docker image & quick example

### We recommend to use docker containers to avoid complicated installation and compilation.

1. Pull the docker image from DockerHub.

```shell
docker pull mingxuangao/dreamdia:v2.0.3
```

2. Enter a container for testing.

```shell
docker run -it --name dreamdia_example --gpus all -v /YOUR/OWN/WORK/PATH:/tmp/work mingxuangao/dreamdia:v2.0.3 /bin/bash
```

* Using the argument `--gpus all` will considerably accelerate the DreamDIA<sup>XMBD</sup> pipeline if you have some GPUs. 

3. Activate the conda environment.

```shell
source activate keras
```

4. Run the example.

```shell
cd /tmp/dreamdia_example
python ../Dream-DIA/DreamDIA.py dreamscore --file_dir raw_data --lib lib.tsv --out example_results
```

The testing results will be in `example_results`.

## 2. Build from source

### Requirements

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
keras-gpu
statsmodels
xgboost
```

If .raw files are going to be fed directly to DreamDIA<sup>XMBD</sup> in Linux systems, [mono](https://www.mono-project.com/download/stable/#download-lin) must be installed.

We recommend to use [Anaconda](https://www.anaconda.com/products/individual#Downloads) to build the environment and install the required libraries as follows.

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
```

### Download
https://github.com/xmuyulab/DreamDIA-XMBD/releases/tag/v2.0.3

### Installation

```bash
cd DreamDIA-XMBD-vXXX
bash build.sh
```

### Run the example

##### (1) download the example data

https://github.com/xmuyulab/DreamDIA-XMBD/tree/main/example_data

##### (2) run

```shell
python PATH/TO/DreamDIA-XMBD-vXXX/DreamDIA.py dreamscore --file_dir example_data/raw_data --lib example_data/lib.tsv --win example_data/win.tsv --out example_data/example_results
```

## 3. Quick start

```bash
python DreamDIA-XMBD-vXXX/DreamDIA.py dreamscore --help
```

```bash
python DreamDIA-XMBD-vXXX/DreamDIA.py dreamscore --file_dir rawdata_dir --lib library.tsv --out output_dir
```

## 4. Notes

#### 1. DIA raw data files

* Centroided .mzML or .mzXML files are supported at any time. 

* If .raw files are going to be fed directly to DreamDIA<sup>XMBD</sup> in Linux systems, [mono](https://www.mono-project.com/download/stable/#download-lin) must be installed first for the data format conversion by ThermoRawFileParser.

* You can also use our docker image to process .raw files directly, while it may take some time for data format conversion.

  ```shell
  # /tmp/dreamdia_example
  python ../Dream-DIA/DreamDIA.py dreamscore --file_dir raw_data_raw --lib lib.tsv --out example_results_raw
  ```

All raw data files should be at the same directory as below. 

```
# rawdata_dir/
	rawdata_1.mzML
	rawdata_2.mzXML
	rawdata_3.raw
```

#### 2. Spectral libraries

Only .tsv libraries are supported. All of the fields required by DreamDIA<sup>XMBD</sup> are listed in `Dream-DIA/lib_col_settings`. Users can modify this file to adjust their own spectral libraries.

#### 3. output

DreamDIA-XMBD outputs peptide and protein identification and quantification results. An empty directory is suggested for the `--out` argument to save all of the output files.

#### 4*. Advanced: train your own deep representation models

See the guidance in `Train_customized_models.ipynb` to train your own deep representation models.

## 5. Cite this article

Gao, M., Yang, W., Li, C. et al. Deep representation features from DreamDIA<sup>XMBD</sup> improve the analysis of data-independent acquisition proteomics. Commun Biol 4, 1190 (2021). https://doi.org/10.1038/s42003-021-02726-6
