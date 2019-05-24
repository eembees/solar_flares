from pathlib import Path
import ijson
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations
from sklearn.preprocessing import maxabs_scale


def decode_obj(line, pos=0, decoder=JSONDecoder()):
    no_white_space_regex = re.compile(r'[^\s]')
    while True:
        match = no_white_space_regex.search(line, pos)
        if not match:
            return
        pos = match.start()
        try:
            obj, pos = decoder.raw_decode(line, pos)
        except JSONDecodeError as err:
            print('Oops! something went wrong. Error: {}'.format(err))
        yield obj


def get_obj_with_last_n_val(line, n):
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    class_label = obj['classNum']

    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    last_n_indices = np.arange(0, 60)[-n:]
    data = data.loc[last_n_indices]

    return {'id': id, 'classType': class_label, 'values': data}


def get_obj_with_all(line):
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    try:
        class_label = obj['classNum']
    except KeyError:
        class_label = None

    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    # last_n_indices = np.arange(0, 60)[-n:]
    # data = data.loc[last_n_indices]

    return {'id': id, 'classType': class_label, 'values': data}


def read_json_data_to_df(file_path: Path):
    """
    Generates a dataframe by concatenating the last values of each
    multi-variate time series. This method is designed as an example
    to show how a json object can be converted into a csv file.
    :param data_dir: the path to the data directory.
    :param file_name: name of the file to be read, with the extension.
    :return: the generated dataframe.
    """

    all_df, labels, ids = [], [], []
    with open(file_path, 'r') as infile:  # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_obj_with_all(line)
            all_df.append(obj['values'])
            labels.append(obj['classType'])
            ids.append(obj['id'])
            print(type(obj))
            print(obj['values'])
            print(type(obj['values']))
            # df =

            exit()

    df = pd.concat(all_df).reset_index(drop=True)
    df = df.assign(LABEL=pd.Series(labels))
    df = df.assign(ID=pd.Series(ids))
    df.set_index([pd.Index(ids)])
    # Uncomment if you want to save this as CSV
    # df.to_csv(file_name + '_last_vals.csv', index=False)
    return df


def read_json_data_to_arr(file_path: Path):
    """
    Generates a dataframe by concatenating the last values of each
    multi-variate time series. This method is designed as an example
    to show how a json object can be converted into a csv file.
    :param data_dir: the path to the data directory.
    :param file_name: name of the file to be read, with the extension.
    :return: the generated dataframe.
    """

    all_df, labels, ids = [], [], []
    with open(file_path, 'r') as infile:  # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_obj_with_all(line)
            # if obj['id'] < 100:
            df = obj['values']
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            df = df.fillna(0.0) # after padding, give up

            all_df.append(df.values)
            labels.append(obj['classType'])
            ids.append(obj['id'])
            # else:
            #     break

    all_df = np.array(all_df)
    labels = np.array(labels)
    ids = np.array(ids)

    return all_df, labels, ids


def save_DF_to_NPZ(fp: Path, out_dir):
    fo = out_dir / fp.with_suffix('.npz').name
    # fo_k = Path(str(fo).replace(('.npz', '_keys.npz')))
    df = pd.read_json(fp, lines=True)

    np.savez(fo, df=df, keys=df.keys, index=df.index)

    pass


def save_arr_to_npz(arr: np.ndarray, labels: np.ndarray, ids: np.ndarray, fo: Path):
    np.savez(fo, data=arr, labels=labels, index=ids)
    pass


def load_npz_file(path: Path):
    a = np.load(path)

    X = a['data']
    try:
        y = a['labels']
    except KeyError:
        y = None
    except ValueError:
        y = None

    return X, y


def save_y_preds(y_index: np.ndarray, y_pred: np.ndarray, fo: Path):
    np.savez(fo, index=y_index, labels=y_pred)
    pass

def preprocess_data(X, scaler=maxabs_scale):
    shap = X.shape
    # print(shap[1:])
    if shap[1:] != (60, 25):
        raise ValueError('Data shape wrong')
    for i, x_i in enumerate(X):
        x_i_t = np.zeros_like(x_i.transpose())
        for j, series in enumerate(x_i.transpose()):
            series = scaler(series)
            x_i_t[j] = series
        X[i] = x_i_t.transpose()
    return X



if __name__ == '__main__':
    data_dir = Path('./input')
    out_dir = data_dir / 'npz'

    file_paths = list(data_dir.glob('*2*.json'))
    print(file_paths)

    for fp in file_paths:
        fo = out_dir / fp.with_suffix('.npz').name
        all_df, labels, ids = read_json_data_to_arr(fp)
        save_arr_to_npz(all_df, labels, ids, fo)
