from sklearn.model_selection import ParameterGrid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Office_Products', help='choose the dataset')
parser.add_argument('--method', type=str, default='pers_page_rank', help='choose the method')
parser.add_argument('--missing_modality', type=str, default='visual')
parser.add_argument('--gpu_id', type=int, default=0, help='choose gpu id')
args = parser.parse_args()

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "--round": [1, 2, 3, 4, 5],
    "--percentage": [10, 50, 90]
})

bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"python impute_percentages.py "
             f"--data {args.data} "
             f"--gpu {args.gpu_id} "
             f"--layers {hyp['--layers']} "
             f"--top_k {hyp['--top_k']} "
             f"--missing_modality {args.missing_modality} "
             f"--percentage {hyp['--percentage']} "
             f"--round {hyp['--round']} "
             f"--method heat\n")

with open(f"impute_all_heat_{args.data}_percentages.sh", 'w') as f:
    f.write(bash)
