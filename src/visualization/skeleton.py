"""Basic renderer for (frames, joints, 3) arrays; feature vectors are not joint coordinates."""
import matplotlib.pyplot as plt
def plot_skeleton(sequence, frame=0, ax=None):
    if sequence.ndim!=3 or sequence.shape[-1]!=3: raise ValueError("Expected joint coordinates shaped (frames, joints, 3)")
    ax=ax or plt.subplots()[1]; joints=sequence[frame]
    parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    for i,parent in enumerate(parents):
        if parent>=0: ax.plot([joints[parent,0],joints[i,0]],[joints[parent,2],joints[i,2]],'-o',lw=2)
    ax.set_aspect('equal'); ax.set_title(f'Frame {frame}'); return ax
