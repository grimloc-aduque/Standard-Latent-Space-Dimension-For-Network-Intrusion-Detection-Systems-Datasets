# Autoencoder-based Standard Latent Space for Intrusion Detection Systems Datasets Analytics

# Datasets
* CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
* UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
* NF-UNSW-NB15-v2: https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA1
* CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
* NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

# Data folders

## Full datasets
* datasets folder contains the full dataset with all its features and labels
* datasets_clean folder contains proprocessed datasets, splitted into features (X), binary labels (Y), and multiclass labels (Ym)

## Small datasets
* small_datasets contains files with the first 2000 rows of every dataset
* small_datasets_clean contains the proprocessed small_datasets
small datasets were used for rapid prototyping purposes only

# Python Scripts
* f01_preprocess_datasets.py script preprocesses the datasets
* f02_encoded_classification.py script generates the autoencoders losses employed to determine the optimum dimension for Latent Space
* f03_encoded_classification script generates the comparison results of original vs latent space variations of each classification model in each dataset

# Jupyter Notebooks
* nb01_datasets_exploration.ipynb presents an exploration of features and categories of every dataset
* nb02_autoencoder_loss_analysis.ipynb analizes the loss of the autoencoders and searches for the optimum LS dimension
* nb03_feature_selection.ipynb summarizes the results of the original vs LS comparison and calculate the Q-value for the Wilcoxon Test