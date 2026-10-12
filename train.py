"""Train and evaluate the similarity model.

    python train.py                 # default: 3600 training pairs + 600 unseen-algorithm test pairs
    python train.py --pairs 1200    # faster
"""
import argparse

from simcode.training import run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=3600, help="number of training pairs")
    ap.add_argument("--unseen", type=int, default=600, help="number of unseen-algorithm test pairs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run_training(args.pairs, args.unseen, args.seed)
