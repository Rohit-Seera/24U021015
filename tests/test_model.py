import torch
from src.models.transformer import Pose2TextTransformer
def test_forward_shapes_and_masks():
    m=Pose2TextTransformer(12,20,d_model=32,nhead=4,encoder_layers=1,decoder_layers=1,dim_feedforward=64,dropout=0)
    x=torch.randn(2,5,12);y=torch.tensor([[1,4,2],[1,5,0]]);sm=torch.tensor([[0,0,0,0,0],[0,0,0,1,1]],dtype=torch.bool);tm=y.eq(0)
    out=m(x,y,sm,tm); assert out.shape==(2,3,20) and torch.isfinite(out).all()
def test_generate():
    m=Pose2TextTransformer(4,10,d_model=16,nhead=4,encoder_layers=1,decoder_layers=1,dim_feedforward=32,dropout=0)
    out=m.generate(torch.randn(1,3,4),1,2,6);assert out.shape[0]==1 and out[0,0]==1
