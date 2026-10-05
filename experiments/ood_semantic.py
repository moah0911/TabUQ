"""Semantic OOD: MagicTelescope <-> credit (10-dim same)."""
import json, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_dataset
from models.tabm_wrapper import TabMWrapper
from uq.metrics import compute_all_uq_metrics_classification
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def scores_from_model(model, X_num, X_cat):
    preds = model.predict_ensemble(X_num, X_cat)
    t = torch.from_numpy(preds).float().squeeze(-1)  # (B,k)
    probs2 = torch.stack([1-torch.sigmoid(t), torch.sigmoid(t)], dim=2)
    m = compute_all_uq_metrics_classification(probs2)
    mean_p = probs2.mean(dim=1).numpy()
    return {"entropy": m["predictive_entropy"], "mi": m["mutual_information"], "max_prob": mean_p.max(axis=1)}

def eval_pair(id_ds, ood_ds, seed):
    X_id, C_id, Y_id, info_id = load_dataset(id_ds)
    X_ood, C_ood, Y_ood, info_ood = load_dataset(ood_ds)
    n_num = X_id["test"].shape[1]
    d_out=1
    # cat cards from ID train (no cats for these 10-dim)
    wrapper = TabMWrapper(n_num_features=n_num, cat_cardinalities=None, d_out=d_out, task_type="binclass", k=32, n_blocks=3, d_block=512)
    wrapper.load(str(Path(f"results/models/{id_ds}_seed{seed}.pt")))
    s_id = scores_from_model(wrapper, X_id["test"], C_id["test"])
    s_ood = scores_from_model(wrapper, X_ood["test"], C_ood["test"])
    res={}
    for k in s_id:
        a = -s_id[k] if k=="max_prob" else s_id[k]
        b = -s_ood[k] if k=="max_prob" else s_ood[k]
        y_true=np.concatenate([np.zeros(len(a)), np.ones(len(b))])
        y_score=np.concatenate([a,b])
        auroc=roc_auc_score(y_true, y_score)
        aupr=average_precision_score(y_true, y_score)
        fpr,tpr,_=roc_curve(y_true, y_score)
        idx=np.where(tpr>=0.95)[0]
        fpr95=float(fpr[idx[0]]) if len(idx)>0 else 1.0
        res[k]={"auroc":float(auroc),"aupr":float(aupr),"fpr95":float(fpr95)}
    return res

if __name__=="__main__":
    import argparse, collections
    Path("results/ood").mkdir(parents=True, exist_ok=True)
    pairs=[("MagicTelescope","credit"), ("credit","MagicTelescope")]
    all_res={}
    for id_ds, ood_ds in pairs:
        for seed in [0,1,2]:
            try:
                r=eval_pair(id_ds, ood_ds, seed)
                key=f"{id_ds}_to_{ood_ds}_seed{seed}"
                all_res[key]=r
                print(f"{key}: entropy {r['entropy']['auroc']:.3f} mi {r['mi']['auroc']:.3f}")
                with open(f"results/ood/{key}.json","w") as f:
                    json.dump(r,f,indent=2)
            except Exception as e:
                print(f"FAIL {id_ds}->{ood_ds} s{seed}: {e}")
                import traceback; traceback.print_exc()
    with open("results/ood/semantic_summary.json","w") as f:
        json.dump(all_res,f,indent=2)
    # mean
    by_pair=collections.defaultdict(list)
    for k,v in all_res.items():
        pair="_to_".join(k.split("_to_")[:2]).rsplit("_seed",1)[0]
        # key like MagicTelescope_to_credit_seed0 -> pair MagicTelescope_to_credit
        pair = "_to_".join(k.split("_to_")[:2]).split("_seed")[0]
        # simpler: split
        if "MagicTelescope_to_credit" in k:
            pair="MagicTelescope_to_credit"
        else:
            pair="credit_to_MagicTelescope"
        by_pair[pair].append(v["entropy"]["auroc"])
    for p, vals in by_pair.items():
        print(f"{p}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
