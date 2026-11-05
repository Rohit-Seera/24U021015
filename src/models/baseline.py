"""Weak non-motion-conditioned most-common caption baseline."""
from collections import Counter
class MajorityCaption:
    def fit(self, captions):
        if not captions: raise ValueError("Training captions are required")
        self.caption=Counter(captions).most_common(1)[0][0]; return self
    def predict(self): return self.caption
