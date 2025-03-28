import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_sparse import SparseTensor, mul, sum, fill_diag
import scipy.sparse as sp
import scipy


def get_item_item():
    # compute item_item matrix
    user_item = sp.coo_matrix(([1.0] * len(train), (train[0].tolist(), train[1].tolist())),
                              shape=(train[0].nunique(), num_items_visual), dtype=np.float32)
    item_item = user_item.transpose().dot(user_item).toarray()
    knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=device), args.top_k, dim=-1)
    items_cols = torch.flatten(knn_ind).to(device)
    ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=device)
    items_rows = torch.repeat_interleave(ir, args.top_k).to(device)
    final_adj = SparseTensor(row=items_rows,
                             col=items_cols,
                             value=torch.tensor([1.0] * items_rows.shape[0], device=device),
                             sparse_sizes=(item_item.shape[0], item_item.shape[0]))
    return final_adj


def compute_normalized_laplacian(adj, norm, fill_value=0.):
    adj = fill_diag(adj, fill_value=fill_value)
    deg = sum(adj, dim=-1)
    deg_inv_sqrt = deg.pow_(-norm)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


parser = argparse.ArgumentParser(description="Run imputation.")
parser.add_argument('--data', type=str, default='Beauty')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--method', type=str, default='mean')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--time', type=float, default=5.0)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--percentage', type=int, default=10)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--missing_modality', type=str, default='visual')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

visual_folder = f'data/{args.data}/visual_embeddings_final_indexed_{args.percentage}_{args.round}'
textual_folder = f'data/{args.data}/textual_embeddings_final_indexed_{args.percentage}_{args.round}'

output_visual = f'data/{args.data}/visual_embeddings_final_indexed_{args.method}_{args.percentage}_{args.round}'
output_textual = f'data/{args.data}/textual_embeddings_final_indexed_{args.method}_{args.percentage}_{args.round}'

if args.missing_modality == 'visual':
    missing_visual_indexed = pd.read_csv(os.path.join(f'data/{args.data}', f'missing_visual_indexed_{args.percentage}_{args.round}.tsv'), sep='\t', header=None)
    missing_visual_indexed = set(missing_visual_indexed[0].tolist())
    missing_textual_indexed = set()
elif args.missing_modality == 'textual':
    missing_textual_indexed = pd.read_csv(
        os.path.join(f'data/{args.data}', f'missing_textual_indexed_{args.percentage}_{args.round}.tsv'), sep='\t',
        header=None)
    missing_textual_indexed = set(missing_textual_indexed[0].tolist())
    missing_visual_indexed = set()
else:
    missing_visual_indexed = pd.read_csv(
        os.path.join(f'data/{args.data}', f'missing_visual_indexed_{args.percentage}_{args.round}.tsv'), sep='\t',
        header=None)
    missing_visual_indexed = set(missing_visual_indexed[0].tolist())
    missing_textual_indexed = pd.read_csv(
        os.path.join(f'data/{args.data}', f'missing_textual_indexed_{args.percentage}_{args.round}.tsv'), sep='\t',
        header=None)
    missing_textual_indexed = set(missing_textual_indexed[0].tolist())

if args.method == 'heat':
    if not os.path.exists(output_visual + f'_{args.layers}_{args.top_k}_{args.time}'):
        os.makedirs(output_visual + f'_{args.layers}_{args.top_k}_{args.time}')
    if not os.path.exists(output_textual + f'_{args.layers}_{args.top_k}_{args.time}'):
        os.makedirs(output_textual + f'_{args.layers}_{args.top_k}_{args.time}')
else:
    if not os.path.exists(output_visual):
        os.makedirs(output_visual)
    if not os.path.exists(output_textual):
        os.makedirs(output_textual)

if args.method == 'mean':
    visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
    textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape
    num_items_visual = len(os.listdir(visual_folder))
    num_items_textual = len(os.listdir(textual_folder))

    visual_features = np.empty((num_items_visual, visual_shape[-1])) if num_items_visual else None
    textual_features = np.empty((num_items_textual, textual_shape[-1])) if num_items_textual else None

    if visual_features is not None:
        visual_items = os.listdir(visual_folder)
        for idx, it in enumerate(visual_items):
            visual_features[idx, :] = np.load(os.path.join(visual_folder, it))
        mean_visual = visual_features.mean(axis=0, keepdims=True)
        for miss in missing_visual_indexed:
            np.save(os.path.join(output_visual, f'{miss}.npy'), mean_visual)

    if textual_features is not None:
        textual_items = os.listdir(textual_folder)
        for idx, it in enumerate(textual_items):
            textual_features[idx, :] = np.load(os.path.join(textual_folder, it))
        mean_textual = textual_features.mean(axis=0, keepdims=True)
        for miss in missing_textual_indexed:
            np.save(os.path.join(output_textual, f'{miss}.npy'), mean_textual)

elif args.method == 'heat':

    visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
    textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape

    output_visual = output_visual + f'_{args.layers}_{args.top_k}_{args.time}'
    output_textual = output_textual + f'_{args.layers}_{args.top_k}_{args.time}'

    try:
        train = pd.read_csv(f'data/{args.data}/train_final_indexed.tsv', sep='\t', header=None)
    except FileNotFoundError:
        print('Before imputing through feat_prop, split the dataset into train/val/test!')
        exit()

    num_items_visual = len(missing_visual_indexed) + len(os.listdir(visual_folder))
    num_items_textual = len(missing_textual_indexed) + len(os.listdir(textual_folder))

    visual_features = np.zeros((num_items_visual, visual_shape[-1]))
    textual_features = np.zeros((num_items_textual, textual_shape[-1]))

    adj = get_item_item()

    adj = sp.coo_matrix((adj.storage._value.cpu().numpy(), (adj.storage._row.cpu().numpy(), adj.storage._col.cpu().numpy())), shape=(num_items_visual, num_items_visual))

    num_nodes = num_items_visual
    A_tilde = adj + np.eye(num_nodes)
    D_tilde = 1 / np.sqrt(A_tilde.sum(axis=1))
    D_ = np.zeros((num_items_visual, num_items_visual))
    np.fill_diagonal(D_, D_tilde[0, 0])
    H = D_ @ A_tilde @ D_
    adj = scipy.linalg.expm(-args.time * (np.eye(num_nodes) - H))
    row_idx = np.arange(num_nodes)
    adj[adj.argsort(axis=0)[:num_nodes - args.top_k], row_idx] = 0.
    norm = adj.sum(axis=0)
    norm[norm <= 0] = 1
    adj = adj / norm

    # feat prop on visual features
    for f in os.listdir(visual_folder):
        visual_features[int(f.split('.npy')[0]), :] = torch.from_numpy(
            np.load(os.path.join(visual_folder, f)))

    non_missing_items = list(set(list(range(num_items_visual))).difference(missing_visual_indexed))
    propagated_visual_features = visual_features.copy()

    for idx in range(args.layers):
        print(f'[VISUAL] Propagation layer: {idx + 1}')
        propagated_visual_features = np.matmul(adj, propagated_visual_features)
        propagated_visual_features[non_missing_items] = visual_features[non_missing_items]

    for miss in missing_visual_indexed:
        np.save(os.path.join(output_visual, f'{miss}.npy'), propagated_visual_features[miss])

    # feat prop on textual features
    for f in os.listdir(textual_folder):
        textual_features[int(f.split('.npy')[0]), :] = torch.from_numpy(
            np.load(os.path.join(textual_folder, f)))

    non_missing_items = list(set(list(range(num_items_textual))).difference(missing_textual_indexed))
    propagated_textual_features = textual_features.copy()

    for idx in range(args.layers):
        print(f'[TEXTUAL] Propagation layer: {idx + 1}')
        propagated_textual_features = np.matmul(adj, propagated_textual_features)
        propagated_textual_features[non_missing_items] = textual_features[non_missing_items]

    for miss in missing_textual_indexed:
        np.save(os.path.join(output_textual, f'{miss}.npy'), propagated_textual_features[miss])
