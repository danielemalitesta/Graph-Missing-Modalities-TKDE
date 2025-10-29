import os
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import argparse
from utils.quick_start import quick_start
# os.environ['NUMEXPR_MAX_THREADS'] = '48'

parser = argparse.ArgumentParser(description="Prepare for MMRec.")
parser.add_argument('--data', type=str, default='Office_Products')
parser.add_argument('--model', type=str, default='LGMRec')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--method', type=str, default='heat')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--time', type=float, default=5.0)
parser.add_argument('--top_k', type=int, default=20)
args = parser.parse_args()

root = f'../data/{args.data}'

original_visual = f'{root}/visual_embeddings_indexed'
original_textual = f'{root}/textual_embeddings_indexed'

train = pd.read_csv(f'{root}/train_indexed.tsv', sep='\t', header=None)
val = pd.read_csv(f'{root}/val_indexed.tsv', sep='\t', header=None)
test = pd.read_csv(f'{root}/test_indexed.tsv', sep='\t', header=None)
df = pd.concat([train, val, test])

num_items = df[1].nunique()

visual_embeddings = np.empty((num_items, 2048))
textual_embeddings = np.empty((num_items, 768))

try:
    missing_visual = pd.read_csv(f'{root}/missing_visual_indexed.tsv', sep='\t', header=None)[0].to_list()
except FileNotFoundError:
    missing_visual = set()

try:
    missing_textual = pd.read_csv(f'{root}/missing_textual_indexed.tsv', sep='\t', header=None)[0].to_list()
except FileNotFoundError:
    missing_textual = set()

for it in range(num_items):
    if not it in missing_visual:
        visual_embeddings[it] = np.load(f'{original_visual}/{it}.npy')
    if not it in missing_textual:
        textual_embeddings[it] = np.load(f'{original_textual}/{it}.npy')

if args.method in ['zeros', 'mean', 'random', 'ae']:
    imputed_visual = f'{root}/visual_embeddings_{args.method}_indexed'
    imputed_textual = f'{root}/textual_embeddings_{args.method}_indexed'
    for it in missing_visual:
        visual_embeddings[it] = np.load(f'{imputed_visual}/{it}.npy')
    for it in missing_textual:
        textual_embeddings[it] = np.load(f'{imputed_textual}/{it}.npy')

if args.method in ['gae', 'neigh_mean']:
    imputed_visual = f'{root}/visual_embeddings_{args.method}_{args.top_k}_indexed'
    imputed_textual = f'{root}/textual_embeddings_{args.method}_{args.top_k}_indexed'
    for it in missing_visual:
        visual_embeddings[it] = np.load(f'{imputed_visual}/{it}.npy')
    for it in missing_textual:
        textual_embeddings[it] = np.load(f'{imputed_textual}/{it}.npy')

if args.method == 'feat_prop':
    imputed_visual = f'{root}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'
    imputed_textual = f'{root}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'
    for it in missing_visual:
        visual_embeddings[it] = np.load(f'{imputed_visual}/{it}.npy')
    for it in missing_textual:
        textual_embeddings[it] = np.load(f'{imputed_textual}/{it}.npy')

if args.method == 'heat':
    imputed_visual = f'{root}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_{args.time}_indexed'
    imputed_textual = f'{root}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_{args.time}_indexed'
    for it in missing_visual:
        visual_embeddings[it] = np.load(f'{imputed_visual}/{it}.npy')
    for it in missing_textual:
        textual_embeddings[it] = np.load(f'{imputed_textual}/{it}.npy')

if args.method == 'pers_page_rank':
    imputed_visual = f'{root}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_{args.alpha}_indexed'
    imputed_textual = f'{root}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_{args.alpha}_indexed'
    for it in missing_visual:
        visual_embeddings[it] = np.load(f'{imputed_visual}/{it}.npy')
    for it in missing_textual:
        textual_embeddings[it] = np.load(f'{imputed_textual}/{it}.npy')


config_dict = {
    'gpu_id': 0,
}

quick_start(model=args.model, dataset=args.data, config_dict=config_dict, save_model=True,
            visual_embeddings=visual_embeddings, textual_embeddings=textual_embeddings)