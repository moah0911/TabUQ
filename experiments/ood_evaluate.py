"""OOD evaluation: synthetic noise on TabM (single-model k32, no embeddings)."""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_dataset
from models.tabm_wrapper import TabMWrapper
from uq.metrics import compute_all_uq_metrics_classification, compute_all_uq_metrics_regression
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def ood_scores(model, X_num, X_cat):
    preds = model.predict_ensemble(X_num, X_cat)  # (B,k,d_out) logits
    t = torch.from_numpy(preds).float()
    if model.task_type == "binclass":
        probs = torch.sigmoid(t).squeeze(-1)  # (B,k)
        probs2 = torch.stack([1-probs, probs], dim=2)  # (B,k,2)
        m = compute_all_uq_metrics_classification(probs2)
        entropy = m["predictive_entropy"]
        mi = m["mutual_information"]
        mean_p = probs2.mean(dim=1).numpy()  # (B,2)
        max_prob = mean_p.max(axis=1)
        return {"entropy": entropy, "mi": mi, "max_prob": max_prob}
    else:
        m = compute_all_uq_metrics_regression(t)
        return {"variance": m["predictive_variance"], "std": m["predictive_std"]}

def add_noise(X, sigma_scale, rng):
    # sigma_scale * per-feature train std (passed as std), or raw sigma if std is 1
    std = X.std(axis=0, keepdims=True) + 1e-8
    noise = rng.normal(0, 1, size=X.shape).astype(np.float32) * (sigma_scale * std)
    return X + noise


def apply_mask(X, ratio, rng):
    """Zero-out random fraction of num features per matrix."""
    Xc = X.copy()
    mask = rng.random(X.shape) < ratio
    Xc[mask] = 0.0
    return Xc.astype(np.float32)


def apply_shuffle(X, rng):
    """Permute each column independently (destroys feature correlations)."""
    Xc = X.copy()
    for j in range(X.shape[1]):
        perm = rng.permutation(X.shape[0])
        Xc[:, j] = X[perm, j]
    return Xc.astype(np.float32)


def apply_scale(X, delta):
    """Synthetic covariate shift: uniform scaling mimics population drift."""
    return (X * (1.0 + delta)).astype(np.float32)


def corrupt(X, corruption, rng, sigma=1.0, mask_ratio=0.25, delta=0.3):
    if corruption == "noise":
        return add_noise(X, sigma, rng)
    if corruption == "mask":
        return apply_mask(X, mask_ratio, rng)
    if corruption == "shuffle":
        return apply_shuffle(X, rng)
    if corruption == "scale":
        return apply_scale(X, delta)
    raise ValueError(f"Unknown corruption: {corruption}")


def evaluate_one(dataset, seed, sigma=1.0, corruption="noise", mask_ratio=0.25, delta=0.3):
    # Load dataset
    X_num, X_cat, Y, info = load_dataset(dataset)
    task = info["task_type"]
    x_test = X_num["test"]
    c_test = X_cat["test"]
    # Load model (need n_num etc from train)
    # infer cat_card and d_out like train.py
    n_num = x_test.shape[1]
    d_out = 1
    cat_cards = None
    if c_test is not None:
        vals = np.concatenate([v for v in [X_cat["train"], X_cat["val"], X_cat["test"]] if v is not None], axis=0) if any(v is not None for v in [X_cat["train"], X_cat["val"], X_cat["test"]]) else c_test
        cards = []
        for col in range(vals.shape[1]):
            uniq = np.unique(vals[:, col])
            uniq = uniq[uniq != -1]
            cards.append(int(len(uniq)) if len(uniq) > 0 else 2)
        cat_cards = cards
    # handle multiclass d_out
    if task == "multiclass":
        d_out = int(np.max(Y["train"])+1)
    wrapper = TabMWrapper(n_num_features=n_num, cat_cardinalities=cat_cards, d_out=d_out, task_type=task, k=32, n_blocks=3, d_block=512)
    ckpt = Path(f"results/models/{dataset}_seed{seed}.pt")
    wrapper.load(str(ckpt))
    # ID scores
    id_scores = ood_scores(wrapper, x_test, c_test)
    # OOD scores (corrupted)
    offset = {"noise": 0, "mask": 100, "shuffle": 200, "scale": 300}.get(corruption, 0)
    rng = np.random.default_rng(42+seed+offset)
    x_ood = corrupt(x_test, corruption, rng, sigma=sigma, mask_ratio=mask_ratio, delta=delta)
    ood_sc = ood_scores(wrapper, x_ood, c_test)
    # For classification primary score = entropy, secondary mi, max_prob (negate for roc)
    results = {}
    for key in id_scores:
        # higher entropy/variance/mi = more OOD, higher max_prob = more ID so invert
        if key == "max_prob":
            s_id = -id_scores[key]
            s_ood = -ood_sc[key]
        else:
            s_id = id_scores[key]
            s_ood = ood_sc[key]
        y_true = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
        y_score = np.concatenate([s_id, s_ood])
        try:
            auroc = roc_auc_score(y_true, y_score)
            aupr = average_precision_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            # FPR @ 95% TPR
            idx = np.where(tpr >= 0.95)[0]
            fpr95 = float(fpr[idx[0]]) if len(idx)>0 else 1.0
        except Exception as e:
            auroc = float('nan'); aupr = float('nan'); fpr95 = float('nan')
        results[key] = {"auroc": float(auroc), "aupr": float(aupr), "fpr95": float(fpr95)}
    return results

if __name__ == "__main__":
    import argparse
    from data.loaders import DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=str, default="noise",
                        choices=["noise", "mask", "shuffle", "scale"])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0,1,2])
    args = parser.parse_args()
    if args.corruption == "noise":
        tag = f"sigma{args.sigma}"
    elif args.corruption == "mask":
        tag = f"mask{args.mask_ratio}"
    elif args.corruption == "shuffle":
        tag = "shuffle"
    else:
        tag = f"scale{args.delta}"
    Path("results/ood").mkdir(parents=True, exist_ok=True)
    all_res = {}
    for ds in args.datasets:
        for sd in args.seeds:
            try:
                r = evaluate_one(ds, sd, sigma=args.sigma, corruption=args.corruption,
                                 mask_ratio=args.mask_ratio, delta=args.delta)
                all_res[f"{ds}_seed{sd}"] = r
                print(f"{ds} s{sd} [{args.corruption} {tag}]: entropy AUROC {r.get('entropy',{}).get('auroc', r.get('variance',{}).get('auroc')):.3f}")
                with open(f"results/ood/{ds}_seed{sd}_{tag}.json","w") as f:
                    json.dump(r,f,indent=2)
            except Exception as e:
                print(f"FAIL {ds} s{sd}: {e}")
                import traceback; traceback.print_exc()
    # Summary mean over healthy 8 (exclude churn/credit/Ailerons if collapsed)
    # compute mean entropy AUROC over seeds
    import collections
    by_ds = collections.defaultdict(list)
    for k,v in all_res.items():
        ds=k.rsplit("_seed",1)[0]
        score = v.get("entropy", v.get("variance",{})).get("auroc")
        if score is not None and not np.isnan(score):
            by_ds[ds].append(score)
    for ds, vals in by_ds.items():
        print(f"{ds}: mean AUROC {np.mean(vals):.3f} +- {np.std(vals):.3f} (n={len(vals)})")
    with open(f"results/ood/summary_{tag}.json","w") as f:
        json.dump(all_res, f, indent=2)
