# Autoencoder-based Standard Latent Space for Intrusion Detection Systems Datasets Analytics

Published Paper at: https://ieeexplore.ieee.org/document/10145438

# Raw Datasets
* CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
* UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
* NF-UNSW-NB15-v2: https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA1
* CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
* NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

# Data folders

## Full datasets
* datasets folder contains the full dataset with all its features and labels
* datasets_clean folder contains proprocessed datasets, splitted into features (X), binary labels (Y), and multiclass labels (Ym)
<!--  -->
Run f01_preprocess_datasets.py to generate the datasets_clean files

## Small datasets
* small_datasets contains files with the first 2000 rows of every dataset
* small_datasets_clean contains the proprocessed small_datasets
<!--  -->
small datasets were used for rapid prototyping purposes only

# Python Scripts
* [f01_preprocess_datasets.py](./f01_preprocess_datasets.py) preprocesses the datasets
* [f02_encoded_classification.py](./f02_encoded_classification.py) generates the autoencoders losses employed to determine the optimum dimension for Latent Space
* [f03_encoded_classification.py](./f03_encoded_classification.py) generates the comparison results of original vs latent space variations of each classification model in each dataset
* [f04_encoded_datasets.py](./f04_encoded_datasets.py) script produces a CSV with the dimensionality reduced datasets of dimension 10

# Jupyter Notebooks
* nb01_datasets_exploration.ipynb presents an exploration of features and categories of every dataset
* nb02_autoencoder_loss_analysis.ipynb analizes the loss of the autoencoders and searches for the optimum LS dimension
* nb03_feature_selection.ipynb summarizes the results of the original vs LS comparison and calculate the Q-value for the Wilcoxon Test

# Docker
The python scripts were run inside a docker container within the infraestructure (DGX) of USFQ university
* dockerfile contains the instructions to build the docker image
* requirements.txt includes the python requierements of the project

# Latent Space
The X_encoded folder contains the dimensionality reduced datasets of dimension 10
