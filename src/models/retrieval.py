"""Nearest-motion retrieval baseline using train-only standardized temporal summaries."""
import numpy as np
from src.data.preprocessing import aggregate_motion
class NearestMotion:
    def fit(self, motions, captions, motion_ids):
        return self.fit_features([aggregate_motion(m) for m in motions],captions,motion_ids)
    def fit_features(self, features, captions, motion_ids):
        self.ids=list(motion_ids); self.captions=list(captions); self.features=np.stack(features); self.mean=self.features.mean(0); self.std=np.maximum(self.features.std(0),1e-6); self.features=(self.features-self.mean)/self.std; return self
    def predict(self, motion, query_id=None):
        q=(aggregate_motion(motion)-self.mean)/self.std; dist=np.linalg.norm(self.features-q,axis=1)
        if query_id in self.ids: dist[np.asarray(self.ids)==query_id]=np.inf
        idx=int(np.argmin(dist)); return self.captions[idx],self.ids[idx]
