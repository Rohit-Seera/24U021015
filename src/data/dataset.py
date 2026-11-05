"""HumanML3D loader. One dataset item is a motion-caption pair; references are retained per motion."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def assert_disjoint_splits(train_ids, test_ids):
    overlap=set(train_ids)&set(test_ids)
    if overlap: raise ValueError(f"Train/test ID overlap ({len(overlap)} IDs), e.g. {sorted(overlap)[:3]}")

class HumanML3D:
    def __init__(self, root, motion_dir="new_joint_vecs", max_motion_length=196, normalize=True):
        self.root = Path(root); self.motion_dir = motion_dir; self.max_motion_length = max_motion_length
        if not self.root.exists(): raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.motion_path = self.root / motion_dir
        self.text_path = self.root / "texts"
        if not self.motion_path.is_dir() or not self.text_path.is_dir():
            raise FileNotFoundError(f"Expected {self.motion_path} and {self.text_path}. Extract HumanML3D archives first.")
        self.mean = self._stat("Mean.npy") if normalize else None; self.std = self._stat("Std.npy") if normalize else None

    def _stat(self, name):
        path = self.root / name
        return np.load(path).astype(np.float32) if path.exists() else None

    def split_ids(self, split):
        path = self.root / f"{split}.txt"
        if not path.is_file(): raise FileNotFoundError(f"Missing split list: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def captions(self, motion_id):
        path = self.text_path / f"{motion_id}.txt"
        if not path.is_file(): return []
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("#"); caption = parts[0].strip()
            if not caption: continue
            try:
                start, end = float(parts[-2]), float(parts[-1])
                if start != 0 or end != 0: continue  # HumanML3D annotations can caption subspans; this model uses full clips.
            except (ValueError, IndexError): pass
            result.append(caption)
        return result

    def motion(self, motion_id):
        path = self.motion_path / f"{motion_id}.npy"
        if not path.is_file(): raise FileNotFoundError(path)
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.reshape(arr.shape[0], -1)  # coordinate sequences become per-frame feature vectors for the model
        if arr.ndim != 2 or arr.shape[0] == 0: raise ValueError(f"Invalid motion array {path}: expected (frames, features) or (frames, joints, 3), got {arr.shape}")
        if self.mean is not None and self.std is not None and self.mean.shape == (arr.shape[1],) and self.std.shape == (arr.shape[1],):
            arr = (arr - self.mean) / np.maximum(self.std, 1e-8)
        return arr[:self.max_motion_length]

    def inspect(self, splits=("train", "val", "test")):
        report = {"root": str(self.root), "motion_dir": self.motion_dir, "splits": {}, "feature_dim": None, "issues": []}
        sets = {}
        for split in splits:
            try: ids = self.split_ids(split)
            except FileNotFoundError as exc: report["issues"].append(str(exc)); continue
            sets[split] = set(ids); valid = 0
            for mid in ids:
                try:
                    motion = self.motion(mid); caps = self.captions(mid)
                    if not caps: report["issues"].append(f"No full-sequence captions: {mid}"); continue
                    report["feature_dim"] = motion.shape[1]; valid += 1
                except (OSError, ValueError, EOFError) as exc: report["issues"].append(f"{mid}: {exc}")
            report["splits"][split] = {"ids":len(ids), "valid_pairs":valid}
        if "train" in sets and "test" in sets and sets["train"] & sets["test"]: report["issues"].append("train/test ID overlap")
        return report

class MotionCaptionDataset(Dataset):
    def __init__(self, loader, ids, vocab, max_caption_length=40, training=False):
        self.loader, self.vocab, self.max_caption_length, self.training = loader, vocab, max_caption_length, training
        self.examples = []
        self.references = {}
        for mid in ids:
            caps = loader.captions(mid)
            if caps:
                self.references[mid] = caps
                self.examples.extend((mid, cap) for cap in caps)
        if not self.examples: raise ValueError("No motion-caption pairs found for requested split")

    def __len__(self): return len(self.examples)

    def __getitem__(self, index):
        mid, caption = self.examples[index]
        if self.training:
            # Each annotation remains a valid paired training example.
            pass
        return {"id":mid, "motion":torch.from_numpy(self.loader.motion(mid)), "caption":torch.tensor(self.vocab.encode(caption, self.max_caption_length), dtype=torch.long), "text":caption}

def collate_pairs(batch, pad_id=0):
    motions=[b["motion"] for b in batch]; caps=[b["caption"] for b in batch]
    x=torch.zeros(len(batch),max(len(x) for x in motions),motions[0].shape[1]); xm=torch.ones(x.shape[:2],dtype=torch.bool)
    y=torch.full((len(batch),max(map(len,caps))),pad_id,dtype=torch.long); ym=torch.ones_like(y,dtype=torch.bool)
    for i,(motion,cap) in enumerate(zip(motions,caps)):
        x[i,:len(motion)]=motion; xm[i,:len(motion)]=False; y[i,:len(cap)]=cap; ym[i,:len(cap)]=False
    return {"ids":[b["id"] for b in batch],"motions":x,"src_padding_mask":xm,"captions":y,"tgt_padding_mask":ym,"texts":[b["text"] for b in batch]}
