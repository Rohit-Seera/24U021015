"""Transformer encoder-decoder for continuous motion features to token sequences."""
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__(); self.dropout=nn.Dropout(dropout)
        pos=torch.arange(max_len).float().unsqueeze(1); div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe=torch.zeros(max_len,d_model); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:pe[:,1::2].shape[1]])
        self.register_buffer("pe",pe.unsqueeze(0),persistent=False)
    def forward(self,x): return self.dropout(x+self.pe[:,:x.size(1)])

class Pose2TextTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=256, nhead=4, encoder_layers=3, decoder_layers=3, dim_feedforward=1024, dropout=.1, pad_id=0):
        super().__init__(); self.pad_id=pad_id; self.d_model=d_model
        self.input_projection=nn.Linear(input_dim,d_model); self.motion_position=PositionalEncoding(d_model,dropout)
        self.token_embedding=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.text_position=PositionalEncoding(d_model,dropout)
        self.transformer=nn.Transformer(d_model,nhead,encoder_layers,decoder_layers,dim_feedforward,dropout,batch_first=True)
        self.output=nn.Linear(d_model,vocab_size)
    def forward(self, motions, captions, src_padding_mask=None, tgt_padding_mask=None):
        # captions include BOS; caller supplies decoder inputs captions[:, :-1].
        source=self.motion_position(self.input_projection(motions)); target=self.text_position(self.token_embedding(captions)*math.sqrt(self.d_model))
        t=target.size(1); causal=torch.triu(torch.ones(t,t,device=target.device,dtype=torch.bool),diagonal=1)
        hidden=self.transformer(source,target,tgt_mask=causal,src_key_padding_mask=src_padding_mask,tgt_key_padding_mask=tgt_padding_mask,memory_key_padding_mask=src_padding_mask)
        return self.output(hidden)
    @torch.no_grad()
    def generate(self,motions,bos_id,eos_id,max_length=40,src_padding_mask=None):
        self.eval(); seq=torch.full((motions.size(0),1),bos_id,dtype=torch.long,device=motions.device); done=torch.zeros(motions.size(0),dtype=torch.bool,device=motions.device)
        for _ in range(max_length-1):
            logits=self.forward(motions,seq,src_padding_mask=src_padding_mask)
            nxt=logits[:,-1].argmax(-1); nxt=torch.where(done,torch.full_like(nxt,self.pad_id),nxt); seq=torch.cat([seq,nxt[:,None]],1); done|=nxt.eq(eos_id)
            if done.all(): break
        return seq
