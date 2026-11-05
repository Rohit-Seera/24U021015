from src.data.vocabulary import Vocabulary
from src.models.transformer import Pose2TextTransformer
from src.training.checkpoint import save_checkpoint,load_checkpoint
def test_checkpoint_roundtrip(tmp_path):
    m=Pose2TextTransformer(3,5,d_model=8,nhead=2,encoder_layers=1,decoder_layers=1,dim_feedforward=16);v=Vocabulary().build(['walk']);p=tmp_path/'c.pt';save_checkpoint(p,m,None,1,v,{},.2);ck=load_checkpoint(p);assert ck['epoch']==1 and ck['vocab']['itos']==v.itos
