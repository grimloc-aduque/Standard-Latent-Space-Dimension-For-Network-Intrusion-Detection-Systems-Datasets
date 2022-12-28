import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import shutil

import pandas as pd

from f01_preprocess_datasets import datasets
from f02_autoencoders import buildAutoencoder, trainAutoencoder


_data_folder = 'datasets_clean'
_results_folder = 'X_encoded'
_encoding_dim = 10


def _main():
    shutil.rmtree(_results_folder)
    os.makedirs(_results_folder)
    pool = multiprocessing.Pool(processes=5)
    pool.map(_one_dataset_run, datasets)


def _one_dataset_run(dataset):
    print(f'Processing dataset {dataset}...')
    X = pd.read_csv(f'./{_data_folder}/{dataset}/X.csv')
    Ym = pd.read_csv(f'./{_data_folder}/{dataset}/Ym.csv')

    # Encode Features
    original_dim = X.shape[1]
    autoencoder, encoder = buildAutoencoder(original_dim, _encoding_dim)
    trainAutoencoder(autoencoder, X)
    Xenc = encoder.predict(X)

    # Format and Save
    Features = pd.DataFrame(Xenc, columns =  [f"Feature {i+1}" for i in range(_encoding_dim)])
    Category = pd.DataFrame(Ym.idxmax(axis = 1), columns=["Category"])
    data = pd.concat([Features, Category], axis=1)
    data.to_csv(f"./{_results_folder}/{dataset}.csv", index=False)


if __name__ == '__main__':
    _main()