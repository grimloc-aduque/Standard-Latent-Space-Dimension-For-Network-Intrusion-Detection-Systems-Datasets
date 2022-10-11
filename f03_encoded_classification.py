import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import multiprocessing
from time import time
from datetime import datetime

import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from f01_preprocess_datasets import datasets
from f02_autoencoders import buildAutoencoder, trainAutoencoder


# Configuration
models = ['binaryANN', 'binaryET', 'multiANN', 'multiET']
_data_folder = 'datasets_clean'
_results_folder = 'results_encoded_classification'
_encoding_dim = 10
_num_repetitions = 10

def _main():
    _buildFolderStructure()
    pool = multiprocessing.Pool(processes=5)
    pool.map(_one_dataset_run, datasets)


def _buildFolderStructure():
    shutil.rmtree(_results_folder)
    os.makedirs(_results_folder)
    for dataset in datasets:
        os.makedirs(f'./{_results_folder}/{dataset}')
        for model in models:
            os.makedirs(f'./{_results_folder}/{dataset}/{model}')


# Comparison Pipeline of original vs encoded models on one dataset

def _one_dataset_run(dataset):
    dataXY = load_dataset(dataset)
    dataXY = calculate_encoded_features(dataXY)
    for i in range(_num_repetitions):
        _all_models_comparison(dataXY, dataset)

def load_dataset(dataset):
    X = pd.read_csv(f'./{_data_folder}/{dataset}/X.csv')
    Y = pd.read_csv(f'./{_data_folder}/{dataset}/Y.csv')
    Ym = pd.read_csv(f'./{_data_folder}/{dataset}/Ym.csv')
    return train_test_split(X, Y, Ym)

def calculate_encoded_features(dataXY):
    X_train, X_test = dataXY[0:2]
    original_dim = X_train.shape[1]
    autoencoder, encoder = buildAutoencoder(original_dim, _encoding_dim)
    loss = trainAutoencoder(autoencoder, X_train)
    Xenc_train = encoder.predict(X_train)
    Xenc_test = encoder.predict(X_test)
    dataXY.append(Xenc_train)
    dataXY.append(Xenc_test)
    return dataXY

def _all_models_comparison(dataXY, dataset):
    results = _binary_comparison(BinaryANN, dataXY)
    _results_to_csv(results, dataset, 'binaryANN')
    results = _binary_comparison(BinaryET, dataXY)
    _results_to_csv(results, dataset, 'binaryET')
    results = _multiclass_comparison(MulticlassANN, dataXY)
    _results_to_csv(results, dataset, 'multiANN')
    results = _multiclass_comparison(MulticlassET, dataXY)
    _results_to_csv(results, dataset, 'multiET')

def _results_to_csv(df_results, dataset, model):
    timeStamp = time()
    strTimeStamp = datetime.fromtimestamp(timeStamp).strftime('%Y.%m.%d.%H.%M.%S')
    df_results.to_csv(f'./{_results_folder}/{dataset}/{model}/{strTimeStamp}.csv', index = False)


# Comparison of original vs encoded variations of a given model

def _binary_comparison(ModelType, dataXY):
    X_train, X_test, Y_train, Y_test, Ym_train, Ym_test, Xenc_train, Xenc_test = dataXY
    original_dim = X_train.shape[1]
    # Modelo original
    model = ModelType()
    model.build_model(original_dim)
    model.train_model(X_train, Y_train)
    metrics = model.test_model(X_test, Y_test)
    # Modelo encoded
    modelEnc = ModelType()
    modelEnc.build_model(_encoding_dim)
    modelEnc.train_model(Xenc_train, Y_train)
    enc_metrics = modelEnc.test_model(Xenc_test, Y_test)
    # Resultados
    results = np.concatenate((metrics, enc_metrics)).reshape((1,4))
    results = pd.DataFrame(results)
    results.columns = ['Accuracy', 'F1', 'Enc Accuracy', 'Enc F1']
    return results


def _multiclass_comparison(ModelType, dataXY):
    X_train, X_test, Y_train, Y_test, Ym_train, Ym_test, Xenc_train, Xenc_test = dataXY
    original_dim = X_train.shape[1]
    num_outputs = Ym_train.shape[1]
    # Modelo original
    model = ModelType()
    model.build_model(original_dim, num_outputs)
    model.train_model(X_train, Ym_train)
    metrics = model.test_model(X_test, Ym_test)
    # Modelo encoded
    modelEnc = ModelType()
    modelEnc.build_model(_encoding_dim, num_outputs)
    modelEnc.train_model(Xenc_train, Ym_train)
    enc_metrics = modelEnc.test_model(Xenc_test, Ym_test)
    # Resultados
    labels = np.array(Ym_train.columns, ndmin=2)
    results = np.concatenate((labels, metrics, enc_metrics[1:]))
    results = results.transpose()
    df_results = pd.DataFrame(results)
    df_results.columns = ['Category', 'Proportion', 'F1', 'DR', 'Enc F1', 'Enc DR']
    return df_results


# Models

class BinaryANN:
    def build_model(self, num_inputs):
        layers = [
            keras.layers.Dense(units = 8, input_shape=[num_inputs], activation=keras.activations.relu),
            keras.layers.Dense(units = 1, activation=keras.activations.sigmoid)
        ]
        self.model = keras.Sequential(layers)
        self.model.compile(
            optimizer ='adam',
            loss='mean_squared_error'
        )

    def train_model(self, X, Y):
        batch_size = 32
        fit = self.model.fit(X, Y, epochs=10, batch_size=batch_size, verbose=False)
        return fit

    def test_model(self, X, Y):
        Y_pred = np.round(self.model.predict(X)).flatten()
        return _binary_metrics(Y, Y_pred)


class BinaryET:
    def build_model(self, num_inputs):
        self.model = ExtraTreesClassifier(n_estimators=50, criterion='entropy')

    def train_model(self, X, Y):
        fit = self.model.fit(X, Y)
        return fit

    def test_model(self, X, Y):
        Y_pred = self.model.predict(X)
        return _binary_metrics(Y, Y_pred)


class MulticlassANN:
    def build_model(self, num_inputs, num_outputs):
        layer1 = keras.layers.Dense(units=15, input_shape=[num_inputs], activation=keras.activations.relu)
        layer2 = keras.layers.Dense(units=num_outputs, activation=keras.activations.softmax)
        self.model = keras.Sequential([layer1, layer2])
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy'
        )

    def train_model(self, X, Y):
        batch_size = 32
        fit = self.model.fit(X, Y, epochs=10, batch_size=batch_size, verbose=False)
        return fit

    def test_model(self, X, Ym):
        Ym_pred = np.round(self.model.predict(X))
        return _multiclass_metrics(Ym, Ym_pred)


class MulticlassET:
    def build_model(self, num_inputs, num_outputs):
        self.model = ExtraTreesClassifier(n_estimators=100, criterion='entropy')

    def train_model(self, X, Ym):
        fit = self.model.fit(X, Ym)
        return fit

    def test_model(self, X, Ym):
        Ym_pred = np.round(self.model.predict(X))
        return _multiclass_metrics(Ym, Ym_pred)


# Metrics

def _binary_metrics(Y, Y_pred):
    accuracy = accuracy_score(Y, Y_pred)
    f1 = f1_score(Y, Y_pred)
    return np.array([accuracy, f1])

def _multiclass_metrics(Ym, Ym_pred):
    Ym = np.array(Ym)
    unique, counts = np.unique( np.argmax(Ym, axis=1), return_counts=True)
    proportions = np.zeros(Ym.shape[1])
    proportions[unique] = counts/Ym.shape[0]
    dr = np.zeros(Ym.shape[1])
    f1 = f1_score(Ym, Ym_pred, average=None)
    cm = confusion_matrix(np.argmax(Ym, axis=1), np.argmax(Ym_pred, axis=1))
    dr[unique] = cm.diagonal() / cm.sum(axis=1)
    return np.array([proportions, f1, dr])


if __name__ == '__main__':
    _main()


