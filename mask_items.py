import pandas as pd
import random, os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Run mask items.")
parser.add_argument('--data', type=str, default='Baby')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_repeats', type=int, default=5)
parser.add_argument('--percentages', type=str, default='10 30 50 70 90')
parser.add_argument('--missing_modality', type=str, default='textual')
args = parser.parse_args()

data = args.data

seed = args.seed
num_repeats = args.num_repeats
percentages = [int(el) for el in args.percentages.split(' ')]
missing_modality = args.missing_modality

train = pd.read_csv(f'./data/{data}/train_final_indexed.tsv', sep='\t', header=None)
items = list(range(train[1].nunique()))

visual_folder = f'data/{args.data}/visual_embeddings_final_indexed/torch/ResNet50/avgpool'
textual_folder = f'data/{args.data}/textual_embeddings_final_indexed/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

if missing_modality == 'visual':

    for n in range(num_repeats):

        random.seed(seed + n)

        for p in percentages:

            output_visual = f'data/{args.data}/visual_embeddings_final_indexed_{p}_{n + 1}'
            output_textual = f'data/{args.data}/textual_embeddings_final_indexed_{p}_{n + 1}'

            if not os.path.exists(output_visual):
                os.makedirs(output_visual)

            if not os.path.exists(output_textual):
                os.makedirs(output_textual)

            sampled_visual = random.sample(items, int((p / 100) * len(items)))
            sampled_textual = []

            pd.DataFrame([pd.Series(sampled_visual)]).transpose().sort_values(by=0).to_csv(
                f'./data/{data}/missing_visual_indexed_{p}_{n + 1}.tsv', sep='\t', header=None, index=None)

            pd.DataFrame(sampled_textual).to_csv(
                f'./data/{data}/missing_textual_indexed_{p}_{n + 1}.tsv', sep='\t', header=None, index=None)

            for emb in os.listdir(visual_folder):
                id_ = int(emb.split('.')[0])
                if id_ not in sampled_visual: # not missing
                    current_emb = np.load(os.path.join(visual_folder, emb))
                    np.save(os.path.join(output_visual, f'{id_}.npy'), current_emb)

            for emb in os.listdir(textual_folder):
                id_ = int(emb.split('.')[0])
                if id_ not in sampled_textual: # not missing
                    current_emb = np.load(os.path.join(textual_folder, emb))
                    np.save(os.path.join(output_textual, f'{id_}.npy'), current_emb)
elif missing_modality == 'textual':

    for n in range(num_repeats):

        random.seed(seed + n)

        for p in percentages:

            output_visual = f'data/{args.data}/visual_embeddings_final_indexed_{p}_{n + 1}'
            output_textual = f'data/{args.data}/textual_embeddings_final_indexed_{p}_{n + 1}'

            if not os.path.exists(output_visual):
                os.makedirs(output_visual)

            if not os.path.exists(output_textual):
                os.makedirs(output_textual)

            sampled_visual = []
            sampled_textual = random.sample(items, int((p / 100) * len(items)))

            pd.DataFrame(sampled_visual).to_csv(
                f'./data/{data}/missing_visual_indexed_{p}_{n + 1}.tsv', sep='\t', header=None, index=None)

            pd.DataFrame([pd.Series(sampled_textual)]).transpose().sort_values(by=0).to_csv(
                f'./data/{data}/missing_textual_indexed_{p}_{n + 1}.tsv', sep='\t', header=None, index=None)

            for emb in os.listdir(visual_folder):
                id_ = int(emb.split('.')[0])
                if id_ not in sampled_visual:  # not missing
                    current_emb = np.load(os.path.join(visual_folder, emb))
                    np.save(os.path.join(output_visual, f'{id_}.npy'), current_emb)

            for emb in os.listdir(textual_folder):
                id_ = int(emb.split('.')[0])
                if id_ not in sampled_textual:  # not missing
                    current_emb = np.load(os.path.join(textual_folder, emb))
                    np.save(os.path.join(output_textual, f'{id_}.npy'), current_emb)
else:
    for n in range(num_repeats):

        random.seed(seed + n)

        for p in percentages:

            output_visual = f'data/{args.data}/visual_embeddings_final_indexed_{p}_{n + 1}'
            output_textual = f'data/{args.data}/textual_embeddings_final_indexed_{p}_{n + 1}'

            if not os.path.exists(output_visual):
                os.makedirs(output_visual)

            if not os.path.exists(output_textual):
                os.makedirs(output_textual)

            sampled_visual = sampled_textual = random.sample(items, int((p / 100) * len(items)))

            pd.DataFrame([pd.Series(sampled_visual)]).transpose().sort_values(by=0).to_csv(
                f'./data/{data}/missing_visual_indexed_{p}_{n + 1}.tsv', sep='\t', header=None, index=None)

            pd.DataFrame([pd.Series(sampled_textual)]).transpose().sort_values(by=0).to_csv(
                f'./data/{data}/missing_textual_indexed_{p}_{n + 1}.tsv', sep='\t', header=None, index=None)

            for emb in os.listdir(visual_folder):
                id_ = int(emb.split('.')[0])
                if id_ not in sampled_visual: # not missing
                    current_emb = np.load(os.path.join(visual_folder, emb))
                    np.save(os.path.join(output_visual, f'{id_}.npy'), current_emb)

            for emb in os.listdir(textual_folder):
                id_ = int(emb.split('.')[0])
                if id_ not in sampled_textual: # not missing
                    current_emb = np.load(os.path.join(textual_folder, emb))
                    np.save(os.path.join(output_textual, f'{id_}.npy'), current_emb)
