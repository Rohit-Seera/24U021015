"""Multi-reference text metrics. Optional metric failures are explicitly reported."""
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from src.data.vocabulary import tokenize

def score_one(prediction, references):
    refs=[tokenize(x) for x in references if tokenize(x)]; pred=tokenize(prediction)
    if not refs: raise ValueError("At least one reference is required")
    smooth=SmoothingFunction().method1
    result={f"bleu_{n}":float(sentence_bleu(refs,pred,weights=tuple([1/n]*n),smoothing_function=smooth)) for n in range(1,5)}
    scorer=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True); result["rouge_l"]=max(scorer.score(" ".join(r)," ".join(pred))["rougeL"].fmeasure for r in refs)
    try: result["meteor"]=float(meteor_score(refs,pred))
    except LookupError: result["meteor"]=None
    return result

def aggregate_scores(rows):
    keys=list(rows[0]) if rows else []
    return {k:(sum(r[k] for r in rows if isinstance(r.get(k),(int,float)))/sum(isinstance(r.get(k),(int,float)) for r in rows) if any(isinstance(r.get(k),(int,float)) for r in rows) else None) for k in keys}
