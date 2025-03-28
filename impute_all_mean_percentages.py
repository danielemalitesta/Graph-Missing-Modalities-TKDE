from sklearn.model_selection import ParameterGrid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--missing_modality', type=str, default='visual')
args = parser.parse_args()

hyperparams = ParameterGrid({
    "--round": [1, 2, 3, 4, 5],
    "--percentage": [10, 30, 50, 70, 90]
})

bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"CUBLAS_WORKSPACE_CONFIG=:16:8 python impute_percentages.py "
             f"--data {args.data} "
             f"--round {hyp['--round']} "
             f"--method mean "
             f"--missing_modality {args.missing_modality} "
             f"--percentage {hyp['--percentage']}\n")

with open(f"impute_all_mean_{args.data}_percentages.sh", 'w') as f:
    f.write(bash)
