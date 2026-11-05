import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate import evaluate
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/best.pt');p.add_argument('--data-root');p.add_argument('--results-dir',default='results');a=p.parse_args();print(evaluate(a.checkpoint,a.data_root,a.results_dir))
if __name__=='__main__': main()
