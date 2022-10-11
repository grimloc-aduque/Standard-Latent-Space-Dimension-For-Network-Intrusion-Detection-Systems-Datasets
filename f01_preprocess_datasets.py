import pandas as pd
import numpy as np
import multiprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configuration
datasets = ['0_CIC-IDS-2017', '1_UNSW-NB15', '2_NF-UNSW-NB15-v2', '3_CSE-CIC-IDS2018', '4_NSL-KDD']
_input_folder = 'datasets'
_output_folder = 'datasets_clean'

def _main():
    dataset_nums = [0,1,2,3,4]
    pool = multiprocessing.Pool(processes=5)
    pool.map(_clean_and_save_dataset, dataset_nums)

# Procesamiento completo del dataset

def _clean_and_save_dataset(datasetNum):
    dataset = datasets[datasetNum]
    split = _splits[datasetNum]
    print(f'Cleaning dataset {dataset}')
    X, Y, Ym  = _preprocess_dataset(f'./{_input_folder}/{dataset}.csv', split)
    X.to_csv(f'./{_output_folder}/{dataset}/X.csv', index=False)
    Y.to_csv(f'./{_output_folder}/{dataset}/Y.csv', index=False)
    Ym.to_csv(f'./{_output_folder}/{dataset}/Ym.csv', index=False)
    print(f'Finished dataset {dataset}')

def _preprocess_dataset(dataset_path, splitXY):
    data = pd.read_csv(dataset_path)
    X, Y, Ym = splitXY(data)
    Ym = pd.get_dummies(Ym)
    X, Y, Ym = _remove_Nan_Inf(X, Y, Ym)
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    return X,Y,Ym

def _remove_Nan_Inf(X, Y, Ym):
    indexes = ~X.isin([np.nan, np.inf, -np.inf]).any(1)
    return(X.loc[indexes], Y.loc[indexes], Ym.loc[indexes])

# Separacion en Features y Labels

def _encode_nominal_columns(X, nominal_columns):
    for column in nominal_columns:
        X[column] =  LabelEncoder().fit_transform(X[column])
    return X

def _splitXY0(data):
    X = data.drop([' Label'], axis=1)
    Y = (data[' Label'] != 'BENIGN') * 1
    Ym = data[' Label']
    return X, Y, Ym

def _splitXY1(data):
    X = data.drop(['srcip', 'sport', 'dstip', 'dsport', 'ct_ftp_cmd', 'attack_cat', 'Label'], axis=1)
    X = _encode_nominal_columns(X, ['proto', 'state','service'])
    Y = data['Label']
    Ym = data['attack_cat']
    Ym[pd.isna(Ym)] = 'Benign'
    return X, Y, Ym

def _splitXY2(data):
    X = data.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack'], axis=1)
    Y = data['Label']
    Ym = data['Attack']
    return X, Y, Ym

def _splitXY3(data):
    X = data.drop(['Timestamp', 'Label'], axis = 1)
    Y = (data['Label'] != 'Benign') * 1
    Ym = data['Label']
    return X, Y, Ym

def _splitXY4(data):
    X = data.drop(['class', 'difficulty'], axis=1)
    X = _encode_nominal_columns(X, ['protocol_type', 'service', 'flag'])
    Y = (data['class'] != 'normal') * 1
    Ym = data['class']
    return X, Y, Ym

_splits = [_splitXY0, _splitXY1, _splitXY2, _splitXY3, _splitXY4]
    

if __name__ == "__main__":
    _main()