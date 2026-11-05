import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import HumanML3D
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--motion-dir',default='new_joint_vecs');a=p.parse_args()
    print(json.dumps(HumanML3D(a.data_root,a.motion_dir).inspect(),indent=2))
if __name__=='__main__': main()
