import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Prepare for MMRec.")
parser.add_argument('--data', type=str, default='Office_Products')
args = parser.parse_args()

root = f'../data/{args.data}'

if not os.path.exists(f'{root}/{args.data}.inter'):
    train = pd.read_csv(f'{root}/train_indexed.tsv', sep='\t', header=None)
    val = pd.read_csv(f'{root}/val_indexed.tsv', sep='\t', header=None)
    test = pd.read_csv(f'{root}/test_indexed.tsv', sep='\t', header=None)
    train.columns = ['userID', 'itemID', 'rating']
    val.columns = ['userID', 'itemID', 'rating']
    test.columns = ['userID', 'itemID', 'rating']
    train['x_label'] = pd.Series([0] * len(train))
    val['x_label'] = pd.Series([1] * len(val))
    test['x_label'] = pd.Series([2] * len(test))
    df = pd.concat([train, val, test])[['userID', 'itemID', 'x_label']]
    df.to_csv(f'{root}/{args.data}.inter', index=False, sep='\t')