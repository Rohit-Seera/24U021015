"""HumanML3D data is distributed externally; use check_dataset to validate an extracted folder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_dataset import main
if __name__=='__main__': main()
