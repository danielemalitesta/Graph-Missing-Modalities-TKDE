import argparse
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import torch

def get_item_item(top_k):
    # compute item_item matrix
    user_item = sp.coo_matrix(([1.0] * len(train), (train[0].tolist(), train[1].tolist())),
                              shape=(train[0].nunique(), train[1].nunique()), dtype=np.float32)
    item_item = user_item.transpose().dot(user_item).toarray()
    knn_val, knn_ind = torch.topk(torch.tensor(item_item), top_k, dim=-1)
    items_cols = torch.flatten(knn_ind)
    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64)
    items_rows = torch.repeat_interleave(ir, top_k)
    final_adj = sp.coo_array((torch.tensor([1.0] * items_rows.shape[0]).numpy(),
                              (items_rows.numpy(), items_cols.numpy())),
                             (item_item.shape[0], item_item.shape[0]))
    return final_adj

top_k = list(range(10, 110, 10))

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Office_Products', help='choose the dataset')
args = parser.parse_args()

# visual modality

for k in top_k:
    try:
        train = pd.read_csv(f'data/{args.data}/train_final.tsv', sep='\t', header=None)
    except FileNotFoundError:
        print('Before measuring feature homophily, split the dataset into train/val/test!')
        exit()

    items = train[1].unique().tolist()

    visual_folder = f'data/{args.data}/visual_embeddings/torch/ResNet50/avgpool/'
    visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
    visual_features = np.empty((len(items), visual_shape[-1]), dtype=np.float32)
    item_dictionary = {}

    for idx, f in enumerate(items):
        item_dictionary[f] = idx
        visual_features[idx, :] = np.load(os.path.join(visual_folder, f'{f}.npy'))

    users = train[0].unique().tolist()
    user_dictionary = {u: idx for idx, u in enumerate(users)}
    train[0] = train[0].map(user_dictionary)
    train[1] = train[1].map(item_dictionary)
    adj = get_item_item(k)

    avg_ = 0.0
    rows, cols = adj.nonzero()
    for idx in range(adj.nnz):
        avg_ += (visual_features[rows[idx]] + visual_features[cols[idx]])
    avg_ /= (2 * adj.nnz)

    num = 0.0
    den_left = 0.0
    den_right = 0.0
    for idx in range(adj.nnz):
        num += np.dot((visual_features[rows[idx]] - avg_), visual_features[cols[idx]] - avg_)
        den_left += np.dot((visual_features[rows[idx]] - avg_), visual_features[rows[idx]] - avg_)
        den_right += np.dot((visual_features[cols[idx]] - avg_), visual_features[cols[idx]] - avg_)

    h_f = num / np.sqrt(den_left * den_right)

    print(f'Feature homophily on the visual modality @{k}: {h_f}')

# textual modality
for k in top_k:

    train = pd.read_csv(f'data/{args.data}/train_final.tsv', sep='\t', header=None)

    items = train[1].unique().tolist()

    textual_folder = f'data/{args.data}/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'
    textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape
    textual_features = np.empty((len(items), textual_shape[-1]), dtype=np.float32)
    item_dictionary = {}

    for idx, f in enumerate(items):
        item_dictionary[f] = idx
        textual_features[idx, :] = np.load(os.path.join(textual_folder, f'{f}.npy'))

    users = train[0].unique().tolist()
    user_dictionary = {u: idx for idx, u in enumerate(users)}
    train[0] = train[0].map(user_dictionary)
    train[1] = train[1].map(item_dictionary)
    adj = get_item_item(k)

    avg_ = 0.0
    rows, cols = adj.nonzero()
    for idx in range(adj.nnz):
        avg_ += (textual_features[rows[idx]] + textual_features[cols[idx]])
    avg_ /= (2 * adj.nnz)

    num = 0.0
    den_left = 0.0
    den_right = 0.0
    for idx in range(adj.nnz):
        num += np.dot((textual_features[rows[idx]] - avg_), textual_features[cols[idx]] - avg_)
        den_left += np.dot((textual_features[rows[idx]] - avg_), textual_features[rows[idx]] - avg_)
        den_right += np.dot((textual_features[cols[idx]] - avg_), textual_features[cols[idx]] - avg_)

    h_f = num / np.sqrt(den_left * den_right)

    print(f'Feature homophily on the textual modality @{k}: {h_f}')