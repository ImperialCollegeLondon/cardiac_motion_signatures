# Human interpretable signatures of three dimensional cardiac motion traits in health and disease

This repository implements the main steps for the generation of cardiac motion signatures and analysis of clinically relevant phenogroups. It transforms cardiac motion data derived from segmented MRIs into human-interpretable signatures.

<img src="./data/flowchart.png">

## Overview

The implementation is organised as follows:

* Pre-processing of time windows of cardiac data and training of the convolution variational autoencoder - [code/cvae_snapshots.py](code/cvae_snapshots.py)
* Post-processing of temporal signatures and clustering (+ stability analysis) - [code/latent_space_clustering.py](code/latent_space_clustering.py)
* Analysis of statistical enrichment of the phenogroups for demographics, biomarkers, cardiovascular risks and outcomes - [code/stat_enrichment.py](code/stat_enrichment.py)
* Generation of three-dimensional phenogroup motion patterns - [code/dynamism_4d.py](code/dynamism_4d.py)
* Benchmark analysis when swapping time windows for MRI-derived features - [codebenchmark_data.py](codebenchmark_data.py)
* Help modules in [code/utils/](code/utils/)

## Getting Started

To get started with this project, please ensure you have the following prerequisites installed:

- Python 3.8 or higher
- TensorFlow 2.x

### Installation

1. Clone the repository:

```
git clone https://github.com/your-repository/cardiac_motion_signatures.git
```

2. To set up the environment and install the required dependencies using Conda, navigate to the project directory and run:

   ```
   conda env create -f environment.yml
   ```

   This will create a new Conda environment with all the necessary packages installed.
3. Activate the newly created Conda environment:

   ```
   conda activate your_env_name
   ```
