"""LS12 QAT and frozen baseline on fresh seeds."""
import argparse
from pathlib import Path
import torch
import experiment as e
from repair13 import SEEDS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dose',type=int,choices=e.DOSES,required=True);p.add_argument('--arm',choices=('frozen','full'),required=True);a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);e.SEEDS=SEEDS;e.train(a.dose,a.arm,Path('baseline'))
