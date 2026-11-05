"""Clearly labelled toy examples for smoke testing only; never used for benchmark evaluation."""
import numpy as np
def make_toy_data(seed=17):
    rng=np.random.default_rng(seed); templates=["a person walks forward","a person raises both arms","a person turns around","a person steps sideways"]
    rows=[]
    for i in range(24):
        label=i%len(templates); base=np.zeros((12,12),dtype=np.float32); base[:,label]=1; base+=rng.normal(0,.04,base.shape).astype(np.float32)
        rows.append((f"toy{i:03d}",base,templates[label]))
    return rows[:20],rows[20:]
