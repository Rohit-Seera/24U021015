"""Training and validation loops for Pose2Text."""
import csv, logging, random, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import HumanML3D, MotionCaptionDataset, collate_pairs, assert_disjoint_splits
from src.data.vocabulary import Vocabulary
from src.models.transformer import Pose2TextTransformer
from src.training.checkpoint import save_checkpoint
from src.training.checkpoint import load_checkpoint

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, optimizer, device, grad_clip=1.0, train=True):
    model.train(train); loss_fn=nn.CrossEntropyLoss(ignore_index=0); total=0.0; tokens=0
    for batch in loader:
        x=batch["motions"].to(device); y=batch["captions"].to(device); src=batch["src_padding_mask"].to(device); tgt=batch["tgt_padding_mask"][:,:-1].to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits=model(x,y[:,:-1],src,tgt); loss=loss_fn(logits.reshape(-1,logits.size(-1)),y[:,1:].reshape(-1))
            if train: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); optimizer.step()
        n=int(y[:,1:].ne(0).sum()); total+=loss.item()*n; tokens+=n
    return total/max(tokens,1)

def train(config, smoke=False):
    seed_everything(config.get("seed",17)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=config.get("data_root")
    if smoke or not root:
        from src.synthetic import make_toy_data
        train_rows,val_rows=make_toy_data(config.get("seed",17))
        train_texts=[c for _,m,c in train_rows for _ in [0]]; val_texts=[c for _,m,c in val_rows]
        vocab=Vocabulary().build(train_texts); xdim=train_rows[0][1].shape[1]
        class ToyPairs(torch.utils.data.Dataset):
            def __init__(self, rows): self.rows=rows
            def __len__(self): return len(self.rows)
            def __getitem__(self,i):
                mid,motion,caption=self.rows[i]; return {"id":mid,"motion":torch.tensor(motion),"caption":torch.tensor(vocab.encode(caption,config["max_caption_length"])),"text":caption}
        tr,va=ToyPairs(train_rows),ToyPairs(val_rows)
        # Toy collation uses the common implementation.
        tr.references={r[0]:[r[2]] for r in train_rows}; va.references={r[0]:[r[2]] for r in val_rows}
    else:
        loader=HumanML3D(root,config.get("motion_dir","new_joint_vecs"),config.get("max_motion_length",196))
        tr_ids=loader.split_ids("train"); va_ids=loader.split_ids("val")
        assert_disjoint_splits(tr_ids,va_ids)
        if config.get("train_limit"): tr_ids=tr_ids[:int(config["train_limit"])]
        tr_caps=[c for mid in tr_ids for c in loader.captions(mid)]
        vocab=Vocabulary().build(tr_caps)
        tr=MotionCaptionDataset(loader,tr_ids,vocab,config["max_caption_length"],True); va=MotionCaptionDataset(loader,va_ids,vocab,config["max_caption_length"])
        xdim=loader.motion(tr_ids[0]).shape[-1]
    collate=lambda rows: collate_pairs(rows,0)
    tr_dl=DataLoader(tr,batch_size=config["batch_size"],shuffle=True,num_workers=config.get("num_workers",0),collate_fn=collate)
    va_dl=DataLoader(va,batch_size=config["batch_size"],shuffle=False,num_workers=config.get("num_workers",0),collate_fn=collate)
    model=Pose2TextTransformer(xdim,len(vocab.itos),config["d_model"],config["nhead"],config["encoder_layers"],config["decoder_layers"],config["dim_feedforward"],config["dropout"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config.get("weight_decay",0.0)); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
    out=Path(config.get("checkpoint_dir","checkpoints")); out.mkdir(parents=True,exist_ok=True); hist=Path(config.get("results_dir","results")); hist.mkdir(parents=True,exist_ok=True)
    best=float("inf"); stale=0; history=[]; start_epoch=1
    if config.get("resume_from"):
        resumed=load_checkpoint(config["resume_from"])
        if resumed["vocab"]["itos"] != vocab.itos: raise ValueError("Resume checkpoint vocabulary differs from current training vocabulary")
        model.load_state_dict(resumed["model"])
        if resumed.get("optimizer"): optimizer.load_state_dict(resumed["optimizer"])
        best=float(resumed.get("best_val_loss",best)); start_epoch=int(resumed.get("epoch",0))+1
    logger=logging.getLogger("pose2text")
    for epoch in range(start_epoch,config["epochs"]+1):
        tl=run_epoch(model,tr_dl,optimizer,device,config.get("grad_clip",1.0),True); vl=run_epoch(model,va_dl,None,device,train=False); scheduler.step(vl)
        history.append({"epoch":epoch,"train_loss":tl,"validation_loss":vl}); logger.info("epoch %d/%d train=%.4f val=%.4f",epoch,config["epochs"],tl,vl)
        if vl<best:
            best=vl; stale=0; save_checkpoint(out/"best.pt",model,optimizer,epoch,vocab,config,best)
            (out/"vocab.json").write_text(json.dumps(vocab.to_dict(),indent=2),encoding="utf-8")
        else: stale+=1
        if stale>=config.get("patience",8): break
    with (hist/"validation_history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["epoch","train_loss","validation_loss"]);w.writeheader();w.writerows(history)
    from src.evaluation.report import write_reports
    write_reports(hist)
    return model,vocab,history
