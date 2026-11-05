"""Portable checkpoint serialization."""
import torch

def save_checkpoint(path, model, optimizer, epoch, vocab, config, best_val_loss):
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict() if optimizer else None,"epoch":epoch,"vocab":vocab.to_dict(),"config":config,"best_val_loss":best_val_loss},path)

def load_checkpoint(path, map_location="cpu"):
    return torch.load(path,map_location=map_location,weights_only=False)
