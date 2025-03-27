from sklearn.model_selection import ParameterGrid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--gpu_id', type=int, default=0, help='choose gpu id')
parser.add_argument('--missing_modality', type=str, default='visual')
parser.add_argument('--cluster', type=str, default='yes')
args = parser.parse_args()

# hyperparams = ParameterGrid({
#     "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     "--round": [1, 2, 3, 4, 5],
#     "--percentage": [10, 50, 90]
# })

hyperparams = ParameterGrid({
    "--round": [1, 2, 3, 4, 5],
    "--percentage": [10, 30, 50, 70, 90]
})

bash = ''
if args.cluster:
    pass
else:
    bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"CUBLAS_WORKSPACE_CONFIG=:16:8 python impute_autoencoder_percentages.py "
             f"--data {args.data} "
             f"--round {hyp['--round']} "
             f"--percentage {hyp['--percentage']} "
             f"--missing_modality {args.missing_modality} "
             f"--method ae\n")

if args.cluster:
    header = """#!/bin/bash -l
#SBATCH --output=../../slogs/missing-%A_%a.out
#SBATCH --error=../../slogs/missing-%A_%a.err
#SBATCH --partition={1}
#SBATCH --job-name=missing
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB # memory in Mb
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH --time=8:00:00 # time requested in days-hours:minutes:seconds
#SBATCH --array=1-{0}

echo "Setting up bash environment"
source ~/.bashrc
set -x

# Modules
module load conda/4.9.2

cd $HOME/projects/Graph-Missing-Modalities/

# Conda environment
conda activate missing

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
    
"""
    with open(f"impute_all_ae_{args.data}_percentages.sh", 'w') as f:
        f.write(header)
        for idx, command in enumerate(bash.split('\n')[:-1]):
            print(f'test $SLURM_ARRAY_TASK_ID -eq {idx + 1} && sleep 10 && {command} > log_{idx + 1}.log 2>&1', file=f)
else:
    with open(f"impute_all_ae_{args.data}_percentages.sh", 'w') as f:
        f.write(bash)
