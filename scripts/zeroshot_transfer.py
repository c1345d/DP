#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-shot transfer experiment: train on one ETT dataset, test on others.

4 sources x 3 targets x 4 models x 2 variants = 96 evaluations.

Usage:
    python scripts/zeroshot_transfer.py
    python scripts/zeroshot_transfer.py --epochs 15 --seed 42
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

# Import model classes from noise_robustness (same directory)
import noise_robustness as nr

OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCES = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
MODELS = ["DLinear", "NLinear", "TSMixer", "iTransformer"]
SEQ_LEN = 96
PRED_LEN = 96

MODEL_CLASSES = {
    "DLinear": (nr.DLinearModel, nr.DLinear_PC_Model),
    "NLinear": (nr.NLinearModel, nr.NLinear_PC_Model),
    "TSMixer": (nr.TSMixerModel, nr.TSMixer_PC_Model),
    "iTransformer": (nr.iTransformerModel, nr.iTransformer_PC_Model),
}


def load_dataset(name):
    csv_map = {
        "ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv",
        "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
    }
    path = os.path.join(ROOT, "dataset", "forecasting", csv_map.get(name, f"{name}.csv"))
    df = pd.read_csv(path)
    wide = df.pivot(index="date", columns="cols", values="data")
    return wide.values.astype(np.float64)


def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = len(data)
    tr = int(n * train_ratio)
    vl = int(n * (train_ratio + val_ratio))
    return data[:tr], data[tr:vl], data[vl:]


def train_one(model, train_data, val_data, epochs=15, lr=0.001, bs=16, device='cpu'):
    scaler = StandardScaler()
    scaler.fit(train_data)
    tr_n = scaler.transform(train_data)
    vl_n = scaler.transform(val_data)
    tr_X, tr_Y = nr.make_windows(tr_n, SEQ_LEN, PRED_LEN)
    vl_X, vl_Y = nr.make_windows(vl_n, SEQ_LEN, PRED_LEN)

    tr_X_t = torch.FloatTensor(tr_X).to(device)
    tr_Y_t = torch.FloatTensor(tr_Y).to(device)
    vl_X_t = torch.FloatTensor(vl_X).to(device)
    vl_Y_t = torch.FloatTensor(vl_Y).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_X_t))
        for i in range(0, len(tr_X_t), bs):
            idx = perm[i:i + bs]
            pred = model(tr_X_t[idx])
            if pred.shape[1] != PRED_LEN:
                pred = pred[:, -PRED_LEN:, :]
            loss = F.mse_loss(pred, tr_Y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        model.eval()
        with torch.no_grad():
            pred = model(vl_X_t)
            if pred.shape[1] != PRED_LEN:
                pred = pred[:, -PRED_LEN:, :]
            val = F.mse_loss(pred, vl_Y_t).item()

        if val < best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 3:
            break

    if best_state:
        model.load_state_dict(best_state)
    return scaler


@torch.no_grad()
def evaluate(model, target_data, device='cpu'):
    """MSE on target dataset using a target-fitted evaluation scaler."""
    scaler = StandardScaler()
    scaler.fit(target_data)
    t_n = scaler.transform(target_data)
    t_X, t_Y = nr.make_windows(t_n, SEQ_LEN, PRED_LEN)
    X_t = torch.FloatTensor(t_X).to(device)
    Y_t = torch.FloatTensor(t_Y).to(device)
    pred = model(X_t)
    if pred.shape[1] != PRED_LEN:
        pred = pred[:, -PRED_LEN:, :]
    return F.mse_loss(pred, Y_t).item()


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | Epochs: {args.epochs}")
    print(f"Sources: {SOURCES}")
    print(f"Models: {MODELS}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []

    for src in SOURCES:
        print(f"\n{'='*60}")
        print(f"SOURCE: {src}")
        print(f"{'='*60}")

        # Load source data
        src_data = load_dataset(src)
        src_tr, src_vl, _ = split_data(src_data)
        n_channels = src_data.shape[1]
        print(f"  shape={src_data.shape} ch={n_channels}")

        # Pre-load all target test sets
        targets = {t: load_dataset(t) for t in SOURCES if t != src}
        for t, d in targets.items():
            _, _, ts = split_data(d)
            targets[t] = ts
            print(f"  Target {t}: test shape={ts.shape}")

        for model_name in MODELS:
            base_cls, pc_cls = MODEL_CLASSES[model_name]
            print(f"\n  --- {model_name} ---")

            # Train on source
            torch.manual_seed(args.seed)
            base_model = base_cls(SEQ_LEN, PRED_LEN, n_channels).to(device)
            train_one(base_model, src_tr, src_vl, args.epochs, device=device)

            torch.manual_seed(args.seed)
            pc_model = pc_cls(SEQ_LEN, PRED_LEN, n_channels).to(device)
            train_one(pc_model, src_tr, src_vl, args.epochs, device=device)

            # Evaluate on each target
            for tgt, tgt_test in targets.items():
                b_mse = evaluate(base_model, tgt_test, device)
                p_mse = evaluate(pc_model, tgt_test, device)

                results.append({
                    'source': src, 'target': tgt, 'model': model_name,
                    'variant': 'base', 'mse': round(b_mse, 6),
                })
                results.append({
                    'source': src, 'target': tgt, 'model': model_name,
                    'variant': 'pc', 'mse': round(p_mse, 6),
                })
                delta = (p_mse - b_mse) / b_mse * 100
                print(f"    {src}->{tgt}: base={b_mse:.4f} pc={p_mse:.4f} "
                      f"delta={delta:+.1f}% {'GAIN' if delta < 0 else 'LOSS'}")

    return results


def save_and_report(results):
    df = pd.DataFrame(results)

    csv_path = os.path.join(OUTPUT_DIR, "zeroshot_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # Summary table: per (source, model), avg delta% over targets
    print(f"\n{'='*70}")
    print("Zero-Shot Transfer Summary (avg delta% per source->target)")
    print(f"{'='*70}")

    for src in SOURCES:
        print(f"\n  Source: {src}")
        print(f"  {'Model':<14} ", end="")
        targets_list = [t for t in SOURCES if t != src]
        for tgt in targets_list:
            print(f"{'->'+tgt:>14} ", end="")
        print(f"{'Avg':>10}")
        print(f"  {'':->14} " + "-" * (14 * len(targets_list) + 10))

        for m in MODELS:
            print(f"  {m:<14} ", end="")
            deltas = []
            for tgt in targets_list:
                r = df[(df['source'] == src) & (df['target'] == tgt) & (df['model'] == m)]
                b = r[r['variant'] == 'base']['mse']
                p = r[r['variant'] == 'pc']['mse']
                if len(b) > 0 and len(p) > 0:
                    d = (p.values[0] - b.values[0]) / b.values[0] * 100
                    deltas.append(d)
                    flag = "GAIN" if d < 0 else "LOSS"
                    print(f"{d:+6.1f}% {flag[:1]:>4} ", end="")
                else:
                    print(f"{'N/A':>14} ", end="")
            avg = sum(deltas) / len(deltas) if deltas else 0
            print(f"{avg:+7.1f}%")

    # Global summary
    all_pairs = []
    for src in SOURCES:
        for tgt in [t for t in SOURCES if t != src]:
            for m in MODELS:
                r = df[(df['source'] == src) & (df['target'] == tgt) & (df['model'] == m)]
                b = r[r['variant'] == 'base']['mse']
                p = r[r['variant'] == 'pc']['mse']
                if len(b) > 0 and len(p) > 0:
                    d = (p.values[0] - b.values[0]) / b.values[0] * 100
                    all_pairs.append({'source': src, 'target': tgt, 'model': m, 'delta': d})

    n_total = len(all_pairs)
    n_gain = sum(1 for a in all_pairs if a['delta'] < 0)
    avg_d = sum(a['delta'] for a in all_pairs) / n_total
    print(f"\nGlobal: {n_gain}/{n_total} pairs improve ({n_gain/n_total*100:.0f}%)")
    print(f"Average delta: {avg_d:+.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results = run(args)
    save_and_report(results)
    print("\nDone.")


if __name__ == '__main__':
    main()
