import torch
from src.models.transformer import Pose2TextTransformer
def test_autoregressive_has_bos():
    model=Pose2TextTransformer(6,8,d_model=16,nhead=4,encoder_layers=1,decoder_layers=1,dim_feedforward=32)
    result=model.generate(torch.zeros(1,2,6),1,2,5);assert result[0,0].item()==1
