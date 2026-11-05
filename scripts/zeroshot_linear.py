#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-shot transfer: DLinear + NLinear only.
Train on one ETT dataset, test on the other three.
PC corrects via GW-Fusion on target domain, no target training needed.

Usage:
    python scripts/zeroshot_linear.py --epochs 15 --seed 42
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN, PRED_LEN = 96, 96
SOURCES = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
MODELS = ["DLinear", "NLinear"]

# Model definitions (self-contained)

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        pad = self.kernel_size // 2
        front, end = x[:,:,0:1].repeat(1,1,pad), x[:,:,-1:].repeat(1,1,pad)
        out = self.avg(torch.cat([front,x,end], dim=2))
        return out[:,:,:x.shape[2]]


class SeriesDecomp(nn.Module):
    def __init__(self, ks): super().__init__(); self.ma = MovingAvg(ks)
    def forward(self, x): t = self.ma(x); return x - t, t


class DLinearModel(nn.Module):
    def __init__(self, sl, pl, C, individual=False, moving_avg=25):
        super().__init__()
        self.sl, self.pl, self.C = sl, pl, C
        self.individual = individual
        self.decomp = SeriesDecomp(moving_avg)
        if individual:
            self.LS = nn.ModuleList([nn.Linear(sl,pl) for _ in range(C)])
            self.LT = nn.ModuleList([nn.Linear(sl,pl) for _ in range(C)])
        else:
            self.LS = nn.Linear(sl, pl); self.LT = nn.Linear(sl, pl)
    def forward(self, x):
        x = x.permute(0,2,1)
        s, t = self.decomp(x)
        if self.individual:
            so = torch.stack([self.LS[i](s[:,i,:]) for i in range(self.C)], dim=1)
            to = torch.stack([self.LT[i](t[:,i,:]) for i in range(self.C)], dim=1)
        else:
            so = self.LS(s); to = self.LT(t)
        return (so + to).permute(0,2,1)


class BasePCCorrection(nn.Module):
    """Mixin: GM + WK + gated fusion."""
    def gm_alpha(self, xt, xr):
        n = (xt*xr).sum(dim=1); d = (xr*xr).sum(dim=1) + 1e-8; return n/d
    def wk_alpha(self, xt, xr):
        Xt = torch.fft.rfft(xt, dim=1); Xr = torch.fft.rfft(xr, dim=1)
        St = (Xt*Xt.conj()).real; Sr = (Xr*Xr.conj()).real
        Rt = torch.fft.irfft(St, dim=1); Rr = torch.fft.irfft(Sr, dim=1)
        n = (Rt*Rr).sum(dim=1); d = (Rr*Rr).sum(dim=1) + 1e-8
        return torch.sqrt(torch.clamp(n/d, min=0))
    def apply_correction(self, x, xr, yp):
        ag = self.gm_alpha(x, xr); aw = self.wk_alpha(x, xr)
        dc = torch.abs(ag-aw)/(torch.maximum(torch.abs(ag),torch.abs(aw))+1e-6)
        avg = (ag+aw)/2.0; shrink = 1.0 + 0.3*(avg - 1.0)
        alpha = torch.where(dc < 0.15, avg, shrink)
        return yp * alpha.unsqueeze(1)


class DLinear_PC_Model(BasePCCorrection):
    def __init__(self, sl, pl, C, individual=False, moving_avg=25):
        nn.Module.__init__(self)
        self.sl, self.pl, self.C = sl, pl, C
        ol = sl + pl; self.individual = individual
        self.decomp = SeriesDecomp(moving_avg)
        if individual:
            self.LS = nn.ModuleList([nn.Linear(sl,ol) for _ in range(C)])
            self.LT = nn.ModuleList([nn.Linear(sl,ol) for _ in range(C)])
        else:
            self.LS = nn.Linear(sl, ol); self.LT = nn.Linear(sl, ol)
    def forward(self, x):
        B, L = x.shape[0], self.sl
        xt = x.permute(0,2,1); s, t = self.decomp(xt)
        if self.individual:
            so = torch.stack([self.LS[i](s[:,i,:]) for i in range(self.C)], dim=1)
            to = torch.stack([self.LT[i](t[:,i,:]) for i in range(self.C)], dim=1)
        else:
            so = self.LS(s); to = self.LT(t)
        out = (so + to).permute(0,2,1)
        return self.apply_correction(x, out[:,:L,:], out[:,L:,:])


class NLinearModel(nn.Module):
    def __init__(self, sl, pl, C, individual=False):
        super().__init__()
        self.sl, self.pl, self.C = sl, pl, C
        self.individual = individual
        self.Linear = nn.ModuleList([nn.Linear(sl,pl) for _ in range(C)]) if individual else nn.Linear(sl, pl)
    def forward(self, x):
        last = x[:,-1:,:]; xn = x - last; xt = xn.permute(0,2,1)
        if self.individual:
            out = torch.stack([self.Linear[i](xt[:,i,:]) for i in range(self.C)], dim=1)
        else:
            out = self.Linear(xt)
        return out.permute(0,2,1) + last


class NLinear_PC_Model(BasePCCorrection):
    def __init__(self, sl, pl, C, individual=False):
        nn.Module.__init__(self)
        self.sl, self.pl, self.C = sl, pl, C
        ol = sl + pl; self.individual = individual
        self.Linear = nn.ModuleList([nn.Linear(sl,ol) for _ in range(C)]) if individual else nn.Linear(sl, ol)
    def forward(self, x):
        B, L = x.shape[0], self.sl
        last = x[:,-1:,:]; xn = x - last; xt = xn.permute(0,2,1)
        if self.individual:
            out = torch.stack([self.Linear[i](xt[:,i,:]) for i in range(self.C)], dim=1)
        else:
            out = self.Linear(xt)
        out = out.permute(0,2,1) + last
        return self.apply_correction(x, out[:,:L,:], out[:,L:,:])


MODEL_CLASSES = {
    "DLinear": (DLinearModel, DLinear_PC_Model),
    "NLinear": (NLinearModel, NLinear_PC_Model),
}

# Data & training (shared logic)

def load_dataset(name):
    csv_map = {"ETTh1":"ETTh1.csv","ETTh2":"ETTh2.csv","ETTm1":"ETTm1.csv","ETTm2":"ETTm2.csv"}
    path = os.path.join(ROOT, "dataset", "forecasting", csv_map[name])
    df = pd.read_csv(path)
    return df.pivot(index="date", columns="cols", values="data").values.astype(np.float64)


def split_data(data, tr=0.6, vl=0.2):
    n = len(data); a = int(n*tr); b = int(n*(tr+vl))
    return data[:a], data[a:b], data[b:]


def make_windows(data, sl, pl):
    X, Y = [], []
    for i in range(len(data)-sl-pl+1):
        X.append(data[i:i+sl]); Y.append(data[i+sl:i+sl+pl])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_one(model, tr_d, vl_d, epochs=15, lr=0.001, bs=16, device='cpu'):
    sc = StandardScaler(); sc.fit(tr_d)
    tr_n, vl_n = sc.transform(tr_d), sc.transform(vl_d)
    tr_X, tr_Y = make_windows(tr_n, SEQ_LEN, PRED_LEN)
    vl_X, vl_Y = make_windows(vl_n, SEQ_LEN, PRED_LEN)
    tr_Xt = torch.FloatTensor(tr_X).to(device); tr_Yt = torch.FloatTensor(tr_Y).to(device)
    vl_Xt = torch.FloatTensor(vl_X).to(device); vl_Yt = torch.FloatTensor(vl_Y).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    best_val, best_st, pat = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_Xt))
        for i in range(0, len(tr_Xt), bs):
            idx = perm[i:i+bs]; pred = model(tr_Xt[idx])
            if pred.shape[1] != PRED_LEN: pred = pred[:,-PRED_LEN:,:]
            loss = F.mse_loss(pred, tr_Yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pred = model(vl_Xt)
            if pred.shape[1] != PRED_LEN: pred = pred[:,-PRED_LEN:,:]
            val = F.mse_loss(pred, vl_Yt).item()
        if val < best_val:
            best_val = val; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
        if pat >= 3: break
    if best_st: model.load_state_dict(best_st)
    return sc


@torch.no_grad()
def evaluate(model, tgt_data, device='cpu'):
    sc = StandardScaler(); sc.fit(tgt_data)
    X, Y = make_windows(sc.transform(tgt_data), SEQ_LEN, PRED_LEN)
    Xt = torch.FloatTensor(X).to(device); Yt = torch.FloatTensor(Y).to(device)
    pred = model(Xt)
    if pred.shape[1] != PRED_LEN: pred = pred[:,-PRED_LEN:,:]
    return F.mse_loss(pred, Yt).item()


# Main

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Zero-Shot Transfer: {MODELS} on {SOURCES}")
    print(f"Device: {device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("="*60)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    results = []

    for src in SOURCES:
        print(f"\n{'='*60}\nSOURCE: {src}")
        src_data = load_dataset(src); C = src_data.shape[1]
        src_tr, src_vl, _ = split_data(src_data)
        targets = {t: split_data(load_dataset(t))[2] for t in SOURCES if t != src}
        print(f"  shape={src_data.shape} C={C}")

        for mn in MODELS:
            bc, pc = MODEL_CLASSES[mn]
            print(f"  --- {mn} ---")

            torch.manual_seed(args.seed)
            bm = bc(SEQ_LEN, PRED_LEN, C).to(device); train_one(bm, src_tr, src_vl, args.epochs, device=device)
            torch.manual_seed(args.seed)
            pm = pc(SEQ_LEN, PRED_LEN, C).to(device); train_one(pm, src_tr, src_vl, args.epochs, device=device)

            for tgt, ts in targets.items():
                bmse = evaluate(bm, ts, device); pmse = evaluate(pm, ts, device)
                d = (pmse - bmse) / bmse * 100
                results.append({'source':src,'target':tgt,'model':mn,'variant':'base','mse':round(bmse,6)})
                results.append({'source':src,'target':tgt,'model':mn,'variant':'pc','mse':round(pmse,6)})
                print(f"    {src}->{tgt}: base={bmse:.4f} pc={pmse:.4f} d={d:+.1f}%")
    return results


def save_and_report(results):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "zeroshot_linear.csv"), index=False)

    print(f"\n{'='*70}")
    print("Zero-Shot Transfer: DLinear + NLinear")
    for src in SOURCES:
        tgts = [t for t in SOURCES if t != src]
        print(f"\n  Source: {src}")
        print(f"  {'Model':<14} ", end="")
        for t in tgts: print(f"{'->'+t:>14} ", end="")
        print("  Avg")
        for m in MODELS:
            print(f"  {m:<14} ", end="")
            ds = []
            for t in tgts:
                r = df[(df.source==src)&(df.target==t)&(df.model==m)]
                b = r[r.variant=='base']['mse']; p = r[r.variant=='pc']['mse']
                if len(b) and len(p):
                    d = (p.values[0]-b.values[0])/b.values[0]*100; ds.append(d)
                    print(f"{d:+6.1f}%      ", end="")
            print(f"{sum(ds)/len(ds):+6.1f}%" if ds else "  N/A")

    all_d = []
    for src in SOURCES:
        for t in [t for t in SOURCES if t != src]:
            for m in MODELS:
                r = df[(df.source==src)&(df.target==t)&(df.model==m)]
                b = r[r.variant=='base']['mse']; p = r[r.variant=='pc']['mse']
                if len(b) and len(p): all_d.append((p.values[0]-b.values[0])/b.values[0]*100)
    print(f"\nOverall: {sum(1 for d in all_d if d<0)}/{len(all_d)} improve  avg={sum(all_d)/len(all_d):+.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    results = run(args)
    save_and_report(results)
    print("\nDone.")


if __name__ == '__main__':
    main()
