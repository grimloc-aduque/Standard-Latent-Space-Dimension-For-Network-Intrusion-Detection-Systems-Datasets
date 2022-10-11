import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing

import pandas as pd
import numpy as np
from tensorflow import keras

from f01_preprocess_datasets import datasets

# Configuration
_data_folder = 'datasets_clean'
_results_folder = 'results_autoencoders'
_num_repetitions = 10

def _main():
    pool = multiprocessing.Pool(processes=5)
    pool.map(_one_dataset_run, datasets)


def _one_dataset_run(dataset):
    print(f'Processing dataset {dataset}')
    X = pd.read_csv(f'./{_data_folder}/{dataset}/X.csv')
    loss_results = []
    for i in range(_num_repetitions):
        loss = _calculate_loss_over_all_dims(X)
        loss_results.append(loss)
    _save_results(dataset, loss_results)
    print(f'Finished dataset {dataset}')


def _calculate_loss_over_all_dims(X):
    print(f'Searching best encoding dim...')
    losses = []
    maxDim = int(X.shape[1])
    dims = np.arange(1, maxDim + 1)
    for dim in dims:
        autoencoder, encoder = buildAutoencoder(X.shape[1], dim)
        print(f'Training autoencoder with encoding dim {dim}...')
        loss = trainAutoencoder(autoencoder, X)
        losses.append(loss)
    return(losses)


def _save_results(dataset, loss_results):
    loss_results = np.array(loss_results)
    nloss, ndims = loss_results.shape
    dims = np.array(np.arange(1, ndims + 1, dtype=int), ndmin=2)
    results = np.concatenate((dims, loss_results)).transpose()
    results = pd.DataFrame(results)
    # Include columns on results data frame
    columns = ['dims']
    for i in range(1, nloss + 1):
        columns.append(f'loss{i}')
    results.columns = columns
    results.to_csv(f'./{_results_folder}/{dataset}.csv', index = False)


# AUTOENCODERS

def buildAutoencoder(original_dim, encoding_dim):
    inputs = keras.Input(shape=(original_dim, ))
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(inputs)
    decoded = keras.layers.Dense(original_dim, activation='sigmoid')(encoded)
    autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    autoencoder.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    return (autoencoder, encoder)


def trainAutoencoder(autoencoder, X):
    batch_size = 64
    fit = autoencoder.fit(X, X, epochs=50, batch_size=batch_size, shuffle=True, verbose=False)
    return fit.history['loss'][-1]


if __name__ == "__main__":
    _main()
