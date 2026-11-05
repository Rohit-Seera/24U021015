"""Training-caption vocabulary with explicit sequence boundary tokens."""
from collections import Counter
import re

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())

class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.itos = [PAD, BOS, EOS, UNK]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.min_freq = min_freq

    def build(self, texts):
        counts = Counter(token for text in texts for token in tokenize(text))
        for token, count in sorted(counts.items()):
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos); self.itos.append(token)
        return self

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        ids = [self.stoi[BOS]] + [self.stoi.get(t, self.stoi[UNK]) for t in tokenize(text)] + [self.stoi[EOS]]
        return ids[:max_length] if max_length else ids

    def decode(self, ids) -> str:
        words = []
        for idx in ids:
            token = self.itos[int(idx)] if 0 <= int(idx) < len(self.itos) else UNK
            if token == EOS: break
            if token not in (PAD, BOS): words.append(token)
        return " ".join(words)

    def to_dict(self): return {"itos": self.itos, "min_freq": self.min_freq}

    @classmethod
    def from_dict(cls, obj):
        vocab = cls(int(obj.get("min_freq", 1))); vocab.itos = list(obj["itos"]); vocab.stoi = {t:i for i,t in enumerate(vocab.itos)}; return vocab
