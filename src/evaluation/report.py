"""Create plots and readable caption inspection report from saved artifacts."""
import csv
from pathlib import Path

def write_reports(results_dir):
    root=Path(results_dir); pred=root/'predictions.csv'
    if pred.exists():
        with pred.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        if rows:
            with (root/'caption_inspection.txt').open('w',encoding='utf-8') as f:
                f.write('Generated captions and references (first 20 test examples)\n\n')
                for row in rows[:20]:
                    f.write(f"Motion: {row['motion_id']}\nTransformer: {row['transformer']}\nReferences: {row['references']}\n\n")
    try:
        import matplotlib.pyplot as plt
        comp=root/'model_comparison.csv'
        if comp.exists():
            with comp.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
            keys=['bleu_1','bleu_2','bleu_3','bleu_4','rouge_l','meteor']
            if rows:
                keys=[k for k in keys if all(row.get(k) not in ('',None) for row in rows)]
                fig,ax=plt.subplots(figsize=(10,5));width=.8/len(rows);x=list(range(len(keys)))
                for i,row in enumerate(rows): ax.bar([v+i*width for v in x],[float(row[k]) for k in keys],width,label=row['model'])
                ax.set_xticks([v+width*(len(rows)-1)/2 for v in x],keys);ax.set_ylim(0,1);ax.set_ylabel('Score');ax.legend();fig.tight_layout();fig.savefig(root/'model_comparison.png',dpi=150);plt.close(fig)
        hist=root/'validation_history.csv'
        if hist.exists():
            with hist.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
            if rows:
                fig,ax=plt.subplots(figsize=(8,4));ax.plot([int(r['epoch']) for r in rows],[float(r['train_loss']) for r in rows],label='train');ax.plot([int(r['epoch']) for r in rows],[float(r['validation_loss']) for r in rows],label='validation');ax.set_xlabel('Epoch');ax.set_ylabel('Token cross-entropy');ax.legend();fig.tight_layout();fig.savefig(root/'loss_history.png',dpi=150);plt.close(fig)
    except ImportError:
        pass
