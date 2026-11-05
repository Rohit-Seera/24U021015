"""Small data utilities shared by scripts and the demo."""
import numpy as np

def aggregate_motion(motion):
    """Concatenate mean, standard deviation, and endpoint delta over time."""
    x=np.asarray(motion,dtype=np.float32)
    return np.concatenate([x.mean(0),x.std(0),x[-1]-x[0]])
