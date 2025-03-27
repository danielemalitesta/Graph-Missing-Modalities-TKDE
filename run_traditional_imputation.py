from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--data', type=str, default='Office_Products')
parser.add_argument('--method', type=str, default='zeros')
parser.add_argument('--model', type=str, default='vbpr')
args = parser.parse_args()

run_experiment(f"config_files/{args.model}_{args.method}_{args.data}.yml")
