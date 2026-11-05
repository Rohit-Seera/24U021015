import argparse, logging, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import train
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--smoke-test',action='store_true');p.add_argument('--overfit',action='store_true');p.add_argument('--resume');a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    with open(a.config,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    if a.overfit: cfg['train_limit']=4; cfg['epochs']=max(cfg['epochs'],50); cfg['patience']=cfg['epochs']
    if a.resume: cfg['resume_from']=a.resume
    train(cfg,smoke=a.smoke_test)
if __name__=='__main__': main()
