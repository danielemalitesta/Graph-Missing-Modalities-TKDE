import os
import pickle as pkl
import pandas as pd
from scipy.sparse import coo_matrix
from Params import args
import numpy as np

root = f'./Datasets/{args.data}'

if not (os.path.exists(f'{root}/trnMat.pkl') and os.path.exists(f'{root}/tstMat.pkl') and os.path.exists(
        f'{root}/valMat.pkl')):
    train = pd.read_csv(f'{root}/train_indexed.tsv', sep='\t', header=None)
    val = pd.read_csv(f'{root}/val_indexed.tsv', sep='\t', header=None)
    test = pd.read_csv(f'{root}/test_indexed.tsv', sep='\t', header=None)
    df = pd.concat([train, val, test])
    num_users = df[0].nunique()
    num_items = df[1].nunique()

    # train
    user_train = train[0].values
    item_train = train[1].values
    trnMat = coo_matrix(([1.0] * user_train.shape[0], (user_train, item_train)),
                        shape=(num_users, num_items), dtype=np.float32)
    with open(f"{root}/trnMat.pkl", "wb") as f:
        pkl.dump(trnMat, f)

    # test
    user_test = test[0].values
    item_test = test[1].values
    tstMat = coo_matrix(([1.0] * user_test.shape[0], (user_test, item_test)),
                        shape=(num_users, num_items), dtype=np.float32)
    with open(f"{root}/tstMat.pkl", "wb") as f:
        pkl.dump(tstMat, f)

    # val
    user_val = val[0].values
    item_val = val[1].values
    valMat = coo_matrix(([1.0] * user_val.shape[0], (user_val, item_val)),
                        shape=(num_users, num_items), dtype=np.float32)
    with open(f"{root}/valMat.pkl", "wb") as f:
        pkl.dump(valMat, f)