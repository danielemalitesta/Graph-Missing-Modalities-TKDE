from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--data', type=str, default='Beauty')
args = parser.parse_args()

run_experiment(f"config_files/percentages_{args.data}.yml")
