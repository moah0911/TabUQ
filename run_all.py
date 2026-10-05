"""Master runner: parallel seed execution via direct subprocess."""
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import DATASETS


def run_single(dataset_name: str, seed: int, config: dict) -> dict:
    """Run one dataset-seed pair via subprocess."""
    cmd_train = [
        sys.executable, "-m", "experiments.train",
        dataset_name,
        "--seed", str(seed),
        "--k", str(config["k"]),
        "--n-blocks", str(config["n_blocks"]),
        "--d-block", str(config["d_block"]),
        "--epochs", str(config["epochs"]),
        "--patience", str(config["patience"]),
        "--eval-every", str(config["eval_every"]),
        "--device", config["device"],
    ]
    if config.get("use_embeddings", False):
        cmd_train.append("--use-embeddings")
    cmd_train.append("--quiet")
    
    env = {
        **dict(subprocess.os.environ),
        "OMP_NUM_THREADS": str(config["threads"]),
        "MKL_NUM_THREADS": str(config["threads"]),
        "OPENBLAS_NUM_THREADS": str(config["threads"]),
    }
    
    start = time.time()
    print(f"[{dataset_name} seed={seed}] START")
    
    r = subprocess.run(cmd_train, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"[{dataset_name} seed={seed}] TRAIN FAILED:\n{r.stderr}")
        return {"dataset": dataset_name, "seed": seed, "status": "train_failed", "error": r.stderr}
    
    cmd_uq = [
        sys.executable, "-m", "experiments.evaluate_uq",
        dataset_name, "--seed", str(seed), "--quiet",
    ]
    r = subprocess.run(cmd_uq, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"[{dataset_name} seed={seed}] UQ FAILED:\n{r.stderr}")
        return {"dataset": dataset_name, "seed": seed, "status": "uq_failed", "error": r.stderr}
    
    elapsed = time.time() - start
    
    uq_path = Path(f"results/raw/{dataset_name}_seed{seed}_uq.json")
    if uq_path.exists():
        with open(uq_path) as f:
            res = json.load(f)
        res["status"] = "success"
        res["elapsed_seconds"] = elapsed
        print(f"[{dataset_name} seed={seed}] SUCCESS in {elapsed:.1f}s")
        return res
    else:
        return {"dataset": dataset_name, "seed": seed, "status": "missing"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--d-block", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-embeddings", action="store_true", help="Use LinearReLUEmbeddings (TabM†)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    
    datasets = args.datasets or DATASETS
    config = {
        "k": args.k, "n_blocks": args.n_blocks, "d_block": args.d_block,
        "epochs": args.epochs, "patience": args.patience, "eval_every": args.eval_every,
        "device": args.device, "threads": args.threads,
        "use_embeddings": args.use_embeddings,
    }
    
    print("=" * 70)
    print(f"TabM-UQ Runner | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: k={args.k}, blocks={args.n_blocks}, d_block={args.d_block}, epochs={args.epochs}")
    emb_str = "LinearReLUEmbeddings (TabM†)" if args.use_embeddings else "NONE (raw features)"
    print(f"Embeddings: {emb_str}")
    print(f"Workers: {args.workers} | Threads/worker: {args.threads}")
    print("=" * 70)
    
    all_results = []
    running = []
    queue = [(d, s) for d in datasets for s in args.seeds]
    
    while queue or running:
        # Start new jobs up to worker limit
        while queue and len(running) < args.workers:
            d, s = queue.pop(0)
            cmd = [sys.executable, "-u", "-m", "experiments.train", d,
                   "--seed", str(s), "--k", str(args.k), "--n-blocks", str(args.n_blocks),
                   "--d-block", str(args.d_block), "--epochs", str(args.epochs),
                   "--patience", str(args.patience), "--eval-every", str(args.eval_every),
                   "--device", args.device]
            if args.use_embeddings:
                cmd.append("--use-embeddings")
            cmd.append("--quiet")
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**dict(subprocess.os.environ),
                     "OMP_NUM_THREADS": str(args.threads),
                     "MKL_NUM_THREADS": str(args.threads),
                     "OPENBLAS_NUM_THREADS": str(args.threads)},
            )
            running.append((d, s, p, time.time()))
            print(f"[{d} seed={s}] START")
        
        # Poll running jobs
        still_running = []
        for d, s, p, t0 in running:
            if p.poll() is not None:
                elapsed = time.time() - t0
                if p.returncode == 0:
                    # Run UQ
                    r = subprocess.run(
                        [sys.executable, "-m", "experiments.evaluate_uq", d, "--seed", str(s), "--quiet"],
                        capture_output=True, text=True,
                        env={**dict(subprocess.os.environ),
                             "OMP_NUM_THREADS": str(args.threads),
                             "MKL_NUM_THREADS": str(args.threads),
                             "OPENBLAS_NUM_THREADS": str(args.threads)},
                    )
                    uq_path = Path(f"results/raw/{d}_seed{s}_uq.json")
                    if r.returncode == 0 and uq_path.exists():
                        with open(uq_path) as f:
                            res = json.load(f)
                        res["status"] = "success"
                        res["elapsed_seconds"] = elapsed
                        all_results.append(res)
                        print(f"[{d} seed={s}] SUCCESS in {elapsed:.1f}s | Acc={res.get('accuracy', res.get('rmse', 0)):.4f}")
                    else:
                        print(f"[{d} seed={s}] UQ FAILED")
                        all_results.append({"dataset": d, "seed": s, "status": "uq_failed"})
                else:
                    print(f"[{d} seed={s}] TRAIN FAILED ({p.returncode})")
                    all_results.append({"dataset": d, "seed": s, "status": "train_failed"})
            else:
                still_running.append((d, s, p, t0))
        running = still_running
        
        if queue or running:
            time.sleep(5)
    
    with open("results/summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    success = sum(1 for r in all_results if r.get("status") == "success")
    print(f"\nDone: {success}/{len(all_results)} succeeded")
    print(f"Summary: results/summary.json")


if __name__ == "__main__":
    main()
