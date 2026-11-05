"""Deterministic test-set generation and model comparison."""
import csv, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.dataset import HumanML3D, MotionCaptionDataset, collate_pairs, assert_disjoint_splits
from src.data.vocabulary import Vocabulary
from src.models.transformer import Pose2TextTransformer
from src.models.baseline import MajorityCaption
from src.models.retrieval import NearestMotion
from src.training.checkpoint import load_checkpoint
from src.evaluation.metrics import score_one, aggregate_scores

def evaluate(checkpoint_path, data_root=None, results_dir="results"):
    ck=load_checkpoint(checkpoint_path); cfg=ck["config"]; root=data_root or cfg["data_root"]
    if not root:
        return evaluate_toy(ck, results_dir)
    loader=HumanML3D(root,cfg.get("motion_dir","new_joint_vecs"),cfg.get("max_motion_length",196)); train_ids=loader.split_ids("train"); test_ids=loader.split_ids("test")
    assert_disjoint_splits(train_ids,test_ids)
    vocab=Vocabulary.from_dict(ck["vocab"]); dim=loader.motion(train_ids[0]).shape[-1]
    model=Pose2TextTransformer(dim,len(vocab.itos),cfg["d_model"],cfg["nhead"],cfg["encoder_layers"],cfg["decoder_layers"],cfg["dim_feedforward"],cfg["dropout"]); model.load_state_dict(ck["model"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    train_ds=MotionCaptionDataset(loader,train_ids,vocab,cfg["max_caption_length"]); test_ds=MotionCaptionDataset(loader,test_ids,vocab,cfg["max_caption_length"])
    from src.data.preprocessing import aggregate_motion
    train_features=[]; train_caps=[]; train_mids=[]
    for mid in train_ids:
        caps=loader.captions(mid)
        if caps:
            train_features.append(aggregate_motion(loader.motion(mid))); train_caps.append(caps[0]); train_mids.append(mid)
    majority=MajorityCaption().fit([c for m in train_ids for c in loader.captions(m)])
    retrieval=NearestMotion().fit_features(train_features,train_caps,train_mids)
    out=Path(results_dir); out.mkdir(parents=True,exist_ok=True); predrows=[]; scores={"transformer":[],"majority":[],"nearest_motion":[]}
    refs_by_id=test_ds.references
    for mid in test_ids:
        refs=refs_by_id.get(mid,[])
        if not refs: continue
        motion=torch.tensor(loader.motion(mid),dtype=torch.float32,device=device)[None,:,:]; mask=torch.zeros((1,motion.shape[1]),dtype=torch.bool,device=device)
        with torch.no_grad(): generated=model.generate(motion,vocab.stoi["<bos>"],vocab.stoi["<eos>"],cfg["max_caption_length"],mask)
        trans=vocab.decode(generated[0].tolist()); maj=majority.predict(); near,neighbor=retrieval.predict(loader.motion(mid),mid)
        predrows.append({"motion_id":mid,"transformer":trans,"majority":maj,"nearest_motion":near,"nearest_motion_id":neighbor,"references":" || ".join(refs)})
        for name,text in [("transformer",trans),("majority",maj),("nearest_motion",near)]: scores[name].append(score_one(text,refs))
    with (out/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(predrows[0]) if predrows else ["motion_id","transformer","references"]);w.writeheader();w.writerows(predrows)
    comp=[]
    for name,values in scores.items():
        means=aggregate_scores(values); comp.append({"model":name,**means})
    with (out/"model_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(comp[0]) if comp else ["model"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(comp)
    unavailable={"bert_score":"Not run; optional package/model weights not installed or downloaded","cider":"Not implemented (optional)"}
    if not comp or all(row.get("meteor") is None for row in comp): unavailable["meteor"]="NLTK WordNet resource required by METEOR is unavailable"
    result={"dataset":"HumanML3D","split":"test","examples":len(predrows),"tokenization":"lowercase alphanumeric regex; whitespace tokens for metrics","metrics":comp,"unavailable":unavailable,"caveat":"Automatic caption metrics are imperfect proxies for semantic correctness."}
    (out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    from src.evaluation.report import write_reports
    write_reports(out)
    return result

def evaluate_toy(ck, results_dir):
    """Run an explicitly synthetic smoke evaluation, isolated from real-data outputs."""
    from src.synthetic import make_toy_data
    from src.models.baseline import MajorityCaption
    from src.models.retrieval import NearestMotion
    cfg=ck["config"]; vocab=Vocabulary.from_dict(ck["vocab"]); train_rows,test_rows=make_toy_data(cfg.get("seed",17))
    dim=train_rows[0][1].shape[-1]
    model=Pose2TextTransformer(dim,len(vocab.itos),cfg["d_model"],cfg["nhead"],cfg["encoder_layers"],cfg["decoder_layers"],cfg["dim_feedforward"],cfg["dropout"]);model.load_state_dict(ck["model"]);model.eval()
    majority=MajorityCaption().fit([r[2] for r in train_rows]); retrieval=NearestMotion().fit([r[1] for r in train_rows],[r[2] for r in train_rows],[r[0] for r in train_rows])
    predrows=[]; scores={"transformer":[],"majority":[],"nearest_motion":[]}
    for mid,motion,ref in test_rows:
        with torch.no_grad(): output=model.generate(torch.tensor(motion)[None],vocab.stoi["<bos>"],vocab.stoi["<eos>"],cfg["max_caption_length"])
        trans=vocab.decode(output[0].tolist()); maj=majority.predict(); near,neighbor=retrieval.predict(motion,mid)
        predrows.append({"motion_id":mid,"transformer":trans,"majority":maj,"nearest_motion":near,"nearest_motion_id":neighbor,"references":ref})
        for name,caption in [("transformer",trans),("majority",maj),("nearest_motion",near)]: scores[name].append(score_one(caption,[ref]))
    out=Path(results_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(predrows[0]));w.writeheader();w.writerows(predrows)
    comp=[{"model":name,**aggregate_scores(values)} for name,values in scores.items()]
    with (out/"model_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(comp[0]));w.writeheader();w.writerows(comp)
    result={"dataset":"synthetic smoke-test toy data","is_benchmark":False,"split":"toy held-out examples","examples":len(predrows),"metrics":comp,"unavailable":{"bert_score":"Not run; optional model dependency not configured","cider":"Not implemented (optional)"},"caveat":"Synthetic smoke metrics are software diagnostics only and must not be reported as HumanML3D results."}
    (out/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    from src.evaluation.report import write_reports
    write_reports(out)
    return result
