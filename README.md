# Pose2Text

Pose2Text is a research project that maps a sequence of 3D human motion features to a natural-language caption. It uses a PyTorch Transformer encoder-decoder and is designed for HumanML3D. This initial release accepts preprocessed motion arrays; it does not extract poses from RGB video.

## Architecture

```mermaid
flowchart LR
  A[HumanML3D motion frames] --> B[Linear feature projection]
  B --> C[Temporal positional encoding]
  C --> D[Transformer encoder]
  D --> E[Transformer decoder with causal mask]
  F[Training-caption vocabulary] --> E
  E --> G[Caption tokens]
```

HumanML3D's official README describes matching ID-named motion and text files, split lists, optional mean/std arrays, `new_joint_vecs` features, and `new_joints` coordinates. This loader uses full-sequence annotations (start and end are both zero), preserves multiple captions as separate paired training examples, and supports either available feature directory. The dimensionality is read from the actual array rather than assumed. See the [official HumanML3D repository](https://github.com/EricGuoICT/HumanML3D) and its README for processing details and terms.

## Dataset

Obtain HumanML3D via the official project instructions; the dataset is not bundled here. Extract at minimum the split text files, `texts/`, and one of `new_joint_vecs/` or `new_joints/`. `Mean.npy` and `Std.npy` are used if present and dimension-compatible. `new_joint_vecs` is the default 263-ish dimensional representation; `new_joints` contains coordinate sequences and has a different dimension.

Expected extracted layout:

```text
HumanML3D/
  texts/000000.txt
  new_joint_vecs/000000.npy   # or new_joints/000000.npy
  train.txt
  val.txt
  test.txt
  Mean.npy                   # optional
  Std.npy                    # optional
```

The source data has its own access and redistribution conditions. Do not redistribute HumanML3D or infer that this code's license grants data rights. Cite Guo et al., *Generating Diverse and Natural 3D Human Motions from Text*, CVPR 2022, and follow the upstream repository's current terms.

## Install and run

Python 3.10+ is recommended. On Windows PowerShell or Linux/macOS:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_dataset.py --data-root data/HumanML3D
python -m pytest -q
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best.pt
streamlit run app.py
```

Set `data_root` and, if needed, `motion_dir` in YAML. Configure batch size, epochs, maximum motion/caption lengths, and model dimensions there. Training automatically selects CUDA when available, clips gradients, reduces learning rate on validation plateaus, and saves the best validation checkpoint. `--overfit` enables a small training subset for debugging. `--smoke-test` selects the synthetic toy path and must not be presented as a real-data experiment.

Quick smoke test (requires installed dependencies):

```bash
python scripts/train.py --config configs/smoke_test.yaml --smoke-test
```

The smoke run writes a toy checkpoint and validation history under `checkpoints/smoke` and `results/smoke`. It does not provide benchmark metrics. Real evaluation writes `results/metrics.json`, `predictions.csv`, and `model_comparison.csv`; validation loss history is written during training. The comparison uses a train-caption majority baseline and nearest-motion retrieval based on train-standardized concatenated temporal mean, standard deviation, and endpoint displacement. Test motions are excluded from retrieval training and same-ID retrieval is explicitly masked.

## Colab

Open `notebooks/Pose2Text_Colab.ipynb` in Colab. It starts with runtime/GPU checks, explains that data is not preloaded, runs synthetic smoke tests, and gives commands for checking, training, evaluating, and saving outputs. For real training, place or mount the extracted data and set `data_root` to its path. A practical first real-data experiment is to set `train_limit: 2000`, `batch_size: 32`, and train for a few epochs on a GPU; this is a subset experiment, not full-dataset training.

## Metrics and limitations

Evaluation calculates BLEU-1 through BLEU-4 with NLTK sentence BLEU and smoothing, ROUGE-L with `rouge-score`, and METEOR with NLTK over all available full-sequence references. Tokenization is lowercase alphanumeric regex; scores are macro-averaged over test motions. If the NLTK WordNet data needed by METEOR is unavailable, its value is recorded as unavailable. BERTScore is reported as not run unless separately implemented with installed model dependencies; CIDEr is optional and currently not implemented. Metric scores are imperfect proxies for semantic correctness. No performance numbers are included until produced by a real evaluation run.

The demo shows a model-generated caption only when a checkpoint exists, references from the selected test motion, and an illustrative feature trace. Skeleton rendering is available only for actual `(frames, joints, 3)` coordinate arrays; 263D feature vectors are not joint coordinates. Video upload and pose extraction are not supported.

## Reproducibility and results

Random seeds are configurable (`seed`); checkpoint includes model, optimizer, epoch, vocabulary, and config. Generated metric files identify real HumanML3D evaluation. BERTScore's optional pretrained model download is not performed automatically. Data, model checkpoint, and dependency licenses remain governed by their respective owners.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | — | — | — | — | — | — |
| Majority caption | — | — | — | — | — | — |
| Nearest motion | — | — | — | — | — | — |

Values above are a template, not measured results. Future work: beam search, coordinate-aware skeleton visualization in the app, BERTScore/CIDEr integration, and richer motion encoders.
