import os
import argparse
import pandas as pd
import json
from scipy.sparse import coo_matrix
import pickle as pkl
import numpy as np

parser = argparse.ArgumentParser(description="Prepare for MMSSL.")
parser.add_argument('--data', type=str, default='Digital_Music')
args = parser.parse_args()

root = f'../data/{args.data}'

train = pd.read_csv(f'{root}/train_indexed.tsv', sep='\t', header=None)
val = pd.read_csv(f'{root}/val_indexed.tsv', sep='\t', header=None)
test = pd.read_csv(f'{root}/test_indexed.tsv', sep='\t', header=None)

if not (os.path.exists(f'{root}/train.json') and os.path.exists(f'{root}/test.json') and os.path.exists(
        f'{root}/val.json')):

    train_json = {}
    val_json = {}
    test_json = {}

    user_train = train[0].unique()
    user_val = val[0].unique()
    user_test = test[0].unique()

    for u in user_train:
        train_json[str(u)] = train[train[0] == u][1].to_list()

    for u in user_val:
        val_json[str(u)] = val[val[0] == u][1].to_list()

    for u in user_test:
        test_json[str(u)] = test[test[0] == u][1].to_list()

    with open(f"{root}/train.json", "w") as f:
        json.dump(train_json, f)

    with open(f"{root}/val.json", "w") as f:
        json.dump(val_json, f)

    with open(f"{root}/test.json", "w") as f:
        json.dump(test_json, f)

if not os.path.exists(f'{root}/trnMat.pkl'):
    df = pd.concat([train, val, test])
    num_users = df[0].nunique()
    num_items = df[1].nunique()
    user_train = train[0].values
    item_train = train[1].values
    trnMat = coo_matrix(([1.0] * user_train.shape[0], (user_train, item_train)),
                        shape=(num_users, num_items), dtype=np.float32)
    with open(f"{root}/trnMat.pkl", "wb") as f:
        pkl.dump(trnMat, f)