"""Load a saved Pose2Text checkpoint and generate one caption for a compatible motion array."""
import numpy as np
import torch
from src.data.vocabulary import Vocabulary
from src.models.transformer import Pose2TextTransformer
from src.training.checkpoint import load_checkpoint

def load_model(checkpoint_path, device=None):
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint=load_checkpoint(checkpoint_path,map_location=device); cfg=checkpoint["config"]
    # Model input dimension is stored by the weight matrix and supports variable source feature formats.
    input_dim=checkpoint["model"]["input_projection.weight"].shape[1]
    vocab=Vocabulary.from_dict(checkpoint["vocab"])
    model=Pose2TextTransformer(input_dim,len(vocab.itos),cfg["d_model"],cfg["nhead"],cfg["encoder_layers"],cfg["decoder_layers"],cfg["dim_feedforward"],cfg["dropout"])
    model.load_state_dict(checkpoint["model"]);model.to(device).eval()
    return model,vocab,device

def generate_caption(motion, model, vocab, device, max_length=40):
    array=np.asarray(motion,dtype=np.float32)
    if array.ndim==3 and array.shape[-1]==3: array=array.reshape(array.shape[0],-1)
    if array.ndim!=2: raise ValueError("Motion must have shape (frames, features) or (frames, joints, 3)")
    expected=model.input_projection.in_features
    if array.shape[1]!=expected: raise ValueError(f"Motion has {array.shape[1]} features; checkpoint expects {expected}")
    with torch.no_grad():
        tokens=model.generate(torch.from_numpy(array)[None].to(device),vocab.stoi["<bos>"],vocab.stoi["<eos>"],max_length)
    return vocab.decode(tokens[0].tolist())
