"""Streamlit interface; predictions are only shown when a trained checkpoint exists."""
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import torch
from src.data.dataset import HumanML3D
from src.data.vocabulary import Vocabulary
from src.models.transformer import Pose2TextTransformer
from src.training.checkpoint import load_checkpoint
from src.evaluation.metrics import score_one
from src.visualization.skeleton import plot_skeleton

st.set_page_config(page_title="Pose2Text",page_icon="🦴",layout="wide")
st.title("Pose2Text")
st.caption("3D motion features → natural-language captions. This demo does not accept RGB video.")
root=st.sidebar.text_input("HumanML3D dataset folder", "data/HumanML3D")
motion_dir=st.sidebar.selectbox("Motion representation", ["new_joint_vecs","new_joints"])
ckpt=Path(st.sidebar.text_input("Checkpoint", "checkpoints/best.pt")); results=Path("results")
if not Path(root).exists(): st.info("Dataset not found. Extract the HumanML3D archives and enter the dataset folder to browse motions.")
else:
    try:
        loader=HumanML3D(root,motion_dir); ids=loader.split_ids("test"); ck=load_checkpoint(ckpt) if ckpt.exists() else None
        mid=st.selectbox("Test motion",ids)
        refs=loader.captions(mid); left,right=st.columns(2)
        with left:
            st.subheader("Motion")
            motion=loader.motion(mid); st.write(f"{len(motion)} frames · {motion.shape[1]} features")
            if motion.shape[1]==66:
                import matplotlib.pyplot as plt
                coords=motion.reshape(-1,22,3); fig,ax=plt.subplots();plot_skeleton(coords,frame=min(len(coords)-1,len(coords)//2),ax=ax);st.pyplot(fig);plt.close(fig)
            else: st.line_chart(motion[:,:min(8,motion.shape[1])])
        with right:
            st.subheader("Caption")
            if ck:
                vocab=Vocabulary.from_dict(ck["vocab"]); cfg=ck["config"]
                model=Pose2TextTransformer(motion.shape[1],len(vocab.itos),cfg['d_model'],cfg['nhead'],cfg['encoder_layers'],cfg['decoder_layers'],cfg['dim_feedforward'],cfg['dropout']);model.load_state_dict(ck['model']);model.eval()
                with torch.no_grad(): pred=model.generate(torch.tensor(motion)[None].float(),vocab.stoi['<bos>'],vocab.stoi['<eos>'],cfg['max_caption_length'])
                caption=vocab.decode(pred[0].tolist());st.success(caption)
                if refs: st.metric("Selected example BLEU-1",f"{score_one(caption,refs)['bleu_1']:.3f}")
            else: st.warning("No trained checkpoint found. Train on the real dataset to generate model captions.")
            st.markdown("**Reference captions**")
            for ref in refs: st.write(f"• {ref}")
        if (results/'metrics.json').exists():
            st.subheader("Saved test-set dashboard")
            st.json(json.loads((results/'metrics.json').read_text(encoding='utf-8')))
        if (results/'predictions.csv').exists(): st.dataframe(pd.read_csv(results/'predictions.csv').head(20),use_container_width=True)
    except Exception as exc: st.error(str(exc))
