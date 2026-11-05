#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise robustness experiment: compare Base vs PC under input perturbation.
Self-contained; no ts_benchmark imports needed.

Global noise: noise patterns are pre-generated per noise level and shared
across all models for fair comparison. 3 global patterns (N_RUNS=3),
results averaged.

5 datasets x 4 models x 2 variants x 3 noise levels = 120 evaluations.

Noise levels: 3%, 6%, 9% of data standard deviation.

Usage:
    python scripts/noise_robustness.py
    python scripts/noise_robustness.py --epochs 10 --datasets ETTh1 --models DLinear
"""

import os, sys, csv, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["ETTh1", "ETTh2", "Solar", "Wind", "Weather"]
MODELS = ["DLinear", "NLinear", "TSMixer", "iTransformer"]
NOISE_LEVELS = [0.03, 0.06, 0.09]  # 3%, 6%, 9% of data std
SEQ_LEN = 96
PRED_LEN = 96
N_RUNS = 3  # different global noise patterns for stability

class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.affine_weight = nn.Parameter(torch.ones(num_features)) if affine else None
        self.affine_bias = nn.Parameter(torch.zeros(num_features)) if affine else None

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight[None, None, :] + self.affine_bias[None, None, :]
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias[None, None, :]) / (self.affine_weight[None, None, :] + self.eps)
            x = x * self.stdev + self.mean
            return x


class MovingAvg(nn.Module):
    """Moving average for trend decomposition."""
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, C, L); manual padding for same-length output
        pad = self.kernel_size // 2
        front = x[:, :, 0:1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        x_padded = torch.cat([front, x, end], dim=2)
        out = self.avg(x_padded)
        # Trim to original length
        return out[:, :, :x.shape[2]]


class SeriesDecomp(nn.Module):
    """Series decomposition into trend + seasonal."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel(nn.Module):
    """DLinear: decomposition + linear layers."""
    def __init__(self, seq_len, pred_len, n_channels, individual=False, moving_avg=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.individual = individual
        self.decomp = SeriesDecomp(moving_avg)
        if individual:
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_channels)])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_channels)])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        x = x.permute(0, 2, 1)  # (B, C, L)
        seasonal, trend = self.decomp(x)
        if self.individual:
            seasonal_out = torch.zeros(x.shape[0], self.n_channels, self.pred_len, device=x.device)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.n_channels):
                seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal)
            trend_out = self.Linear_Trend(trend)
        return (seasonal_out + trend_out).permute(0, 2, 1)  # (B, H, C)


class DLinear_PC_Model(nn.Module):
    """DLinear PC: outputs seq_len+pred_len, splits, applies correction."""
    def __init__(self, seq_len, pred_len, n_channels, individual=False, moving_avg=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        out_len = seq_len + pred_len
        self.individual = individual
        self.decomp = SeriesDecomp(moving_avg)
        if individual:
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, out_len) for _ in range(n_channels)])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, out_len) for _ in range(n_channels)])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, out_len)
            self.Linear_Trend = nn.Linear(seq_len, out_len)

    def gauss_markov_alpha(self, x_true, x_recon):
        """Compute GM correction factor per channel."""
        num = (x_true * x_recon).sum(dim=1)  # (B, C)
        den = (x_recon * x_recon).sum(dim=1) + 1e-8
        return num / den  # (B, C)

    def wiener_khinchin_alpha(self, x_true, x_recon):
        """Compute WK correction via autocorrelation."""
        B, L, C = x_true.shape
        # FFT per channel
        X_true = torch.fft.rfft(x_true, dim=1)
        X_recon = torch.fft.rfft(x_recon, dim=1)
        # Power spectrum
        S_true = (X_true * X_true.conj()).real
        S_recon = (X_recon * X_recon.conj()).real
        # Inverse FFT to autocorrelation
        R_true = torch.fft.irfft(S_true, dim=1)
        R_recon = torch.fft.irfft(S_recon, dim=1)
        # OLS on autocorrelation
        num = (R_true * R_recon).sum(dim=1)
        den = (R_recon * R_recon).sum(dim=1) + 1e-8
        beta = num / den
        return torch.sqrt(torch.clamp(beta, min=0.0))  # sqrt for amplitude ratio

    def forward(self, x):
        B, L, C = x.shape
        x_t = x.permute(0, 2, 1)
        seasonal, trend = self.decomp(x_t)
        out_len = self.seq_len + self.pred_len
        if self.individual:
            seasonal_out = torch.zeros(B, C, out_len, device=x.device)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(C):
                seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal)
            trend_out = self.Linear_Trend(trend)
        out = (seasonal_out + trend_out).permute(0, 2, 1)  # (B, L+H, C)

        x_recon = out[:, :L, :]
        y_pred = out[:, L:, :]

        # GW-Fusion
        alpha_gm = self.gauss_markov_alpha(x, x_recon)  # (B, C)
        alpha_wk = self.wiener_khinchin_alpha(x, x_recon)  # (B, C)

        # Disagreement gate
        delta_c = torch.abs(alpha_gm - alpha_wk) / (torch.maximum(
            torch.abs(alpha_gm), torch.abs(alpha_wk)) + 1e-6)
        avg = (alpha_gm + alpha_wk) / 2.0
        shrink = 1.0 + 0.3 * (avg - 1.0)

        alpha = torch.where(delta_c < 0.15, avg, shrink)
        alpha = alpha.unsqueeze(1)  # (B, 1, C)
        return y_pred * alpha


class NLinearModel(nn.Module):
    """NLinear: subtract last value, linear, add back."""
    def __init__(self, seq_len, pred_len, n_channels, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        if individual:
            self.Linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_channels)])
        else:
            self.Linear = nn.Linear(seq_len, pred_len)
        self.individual = individual

    def forward(self, x):
        # x: (B, L, C)
        last = x[:, -1:, :]
        x_norm = x - last
        x_t = x_norm.permute(0, 2, 1)  # (B, C, L)
        if self.individual:
            out = torch.zeros(x.shape[0], self.n_channels, self.pred_len, device=x.device)
            for i in range(self.n_channels):
                out[:, i, :] = self.Linear[i](x_t[:, i, :])
        else:
            out = self.Linear(x_t)
        out = out.permute(0, 2, 1) + last  # (B, H, C)
        return out


class NLinear_PC_Model(DLinear_PC_Model):
    """NLinear PC: same correction as DLinear PC but with NLinear base."""
    def __init__(self, seq_len, pred_len, n_channels, individual=False):
        nn.Module.__init__(self)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        out_len = seq_len + pred_len
        if individual:
            self.Linear = nn.ModuleList([
                nn.Linear(seq_len, out_len) for _ in range(n_channels)])
        else:
            self.Linear = nn.Linear(seq_len, out_len)
        self.individual = individual

    def forward(self, x):
        B, L, C = x.shape
        last = x[:, -1:, :]
        x_norm = x - last
        x_t = x_norm.permute(0, 2, 1)
        out_len = self.seq_len + self.pred_len
        if self.individual:
            out = torch.zeros(B, C, out_len, device=x.device)
            for i in range(C):
                out[:, i, :] = self.Linear[i](x_t[:, i, :])
        else:
            out = self.Linear(x_t)
        out = out.permute(0, 2, 1) + last

        x_recon = out[:, :L, :]
        y_pred = out[:, L:, :]
        alpha_gm = self.gauss_markov_alpha(x, x_recon)
        alpha_wk = self.wiener_khinchin_alpha(x, x_recon)
        delta_c = torch.abs(alpha_gm - alpha_wk) / (torch.maximum(
            torch.abs(alpha_gm), torch.abs(alpha_wk)) + 1e-6)
        avg = (alpha_gm + alpha_wk) / 2.0
        shrink = 1.0 + 0.3 * (avg - 1.0)
        alpha = torch.where(delta_c < 0.15, avg, shrink)
        return y_pred * alpha.unsqueeze(1)


class TSMixerModel(nn.Module):
    """TSMixer: time-mixing + channel-mixing MLP blocks."""
    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_block=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.revin = RevIN(n_channels)
        self.time_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(seq_len, d_model), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(d_model, seq_len))
            for _ in range(n_block)])
        self.ch_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(n_channels, d_model), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(d_model, n_channels))
            for _ in range(n_block)])
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        x = self.revin(x, 'norm')
        for tm, cm in zip(self.time_mix, self.ch_mix):
            residual = x
            x = x + tm(x.permute(0, 2, 1)).permute(0, 2, 1)  # time mixing
            x = x + cm(x)  # channel mixing
            x = x + residual
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.proj(x).permute(0, 2, 1)  # (B, H, C)
        return self.revin(x, 'denorm')


class TSMixer_PC_Model(DLinear_PC_Model):
    """TSMixer PC: extended projection + correction."""
    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_block=2, dropout=0.1):
        nn.Module.__init__(self)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.revin = RevIN(n_channels)
        self.time_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(seq_len, d_model), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(d_model, seq_len))
            for _ in range(n_block)])
        self.ch_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(n_channels, d_model), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(d_model, n_channels))
            for _ in range(n_block)])
        self.proj = nn.Linear(seq_len, seq_len + pred_len)

    def forward(self, x):
        B, L, C = x.shape
        x = self.revin(x, 'norm')
        for tm, cm in zip(self.time_mix, self.ch_mix):
            residual = x
            x = x + tm(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + cm(x)
            x = x + residual
        x = x.permute(0, 2, 1)
        out = self.proj(x).permute(0, 2, 1)
        out = self.revin(out, 'denorm')

        x_recon = out[:, :L, :]
        y_pred = out[:, L:, :]
        x_denorm = self.revin(x, 'denorm')
        alpha_gm = self.gauss_markov_alpha(x_denorm, x_recon)
        alpha_wk = self.wiener_khinchin_alpha(x_denorm, x_recon)
        delta_c = torch.abs(alpha_gm - alpha_wk) / (torch.maximum(
            torch.abs(alpha_gm), torch.abs(alpha_wk)) + 1e-6)
        avg = (alpha_gm + alpha_wk) / 2.0
        shrink = 1.0 + 0.3 * (avg - 1.0)
        alpha = torch.where(delta_c < 0.15, avg, shrink)
        return y_pred * alpha.unsqueeze(1)


class iTransformerModel(nn.Module):
    """iTransformer: inverted embedding + transformer."""
    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=4,
                 e_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.revin = RevIN(n_channels)
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = self.value_embedding(x.permute(0, 2, 1)) + self.pos_embed  # (B, C, d)
        x = self.encoder(x)  # (B, C, d)
        x = self.proj(x).permute(0, 2, 1)  # (B, H, C)
        return self.revin(x, 'denorm')


class iTransformer_PC_Model(DLinear_PC_Model):
    """iTransformer PC: extended projection + correction."""
    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=4,
                 e_layers=2, d_ff=256, dropout=0.1):
        nn.Module.__init__(self)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.revin = RevIN(n_channels)
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.proj = nn.Linear(d_model, seq_len + pred_len)

    def forward(self, x):
        B, L, C = x.shape
        x_norm = self.revin(x, 'norm')
        emb = self.value_embedding(x_norm.permute(0, 2, 1)) + self.pos_embed
        enc = self.encoder(emb)
        out = self.proj(enc).permute(0, 2, 1)  # (B, L+H, C)
        out = self.revin(out, 'denorm')

        x_recon = out[:, :L, :]
        y_pred = out[:, L:, :]
        alpha_gm = self.gauss_markov_alpha(x, x_recon)
        alpha_wk = self.wiener_khinchin_alpha(x, x_recon)
        delta_c = torch.abs(alpha_gm - alpha_wk) / (torch.maximum(
            torch.abs(alpha_gm), torch.abs(alpha_wk)) + 1e-6)
        avg = (alpha_gm + alpha_wk) / 2.0
        shrink = 1.0 + 0.3 * (avg - 1.0)
        alpha = torch.where(delta_c < 0.15, avg, shrink)
        return y_pred * alpha.unsqueeze(1)


# Model registry
MODEL_CLASSES = {
    "DLinear": (DLinearModel, DLinear_PC_Model),
    "NLinear": (NLinearModel, NLinear_PC_Model),
    "TSMixer": (TSMixerModel, TSMixer_PC_Model),
    "iTransformer": (iTransformerModel, iTransformer_PC_Model),
}


def load_dataset(name):
    csv_map = {"ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv",
               "Solar": "Solar.csv", "Wind": "Wind.csv", "Weather": "Weather.csv"}
    path = os.path.join(ROOT, "dataset", "forecasting", csv_map.get(name, f"{name}.csv"))
    df = pd.read_csv(path)
    wide = df.pivot(index="date", columns="cols", values="data")
    return wide.values.astype(np.float64)


def split_data(data, train_ratio=0.6, val_ratio=0.2):
    n = len(data)
    tr = int(n * train_ratio)
    vl = int(n * (train_ratio + val_ratio))
    return data[:tr], data[tr:vl], data[vl:]


def make_windows(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_one(model, train_data, val_data, epochs=15, lr=0.001, bs=16, device='cpu'):
    scaler = StandardScaler()
    scaler.fit(train_data)
    tr_n = scaler.transform(train_data)
    vl_n = scaler.transform(val_data)
    tr_X, tr_Y = make_windows(tr_n, SEQ_LEN, PRED_LEN)
    vl_X, vl_Y = make_windows(vl_n, SEQ_LEN, PRED_LEN)

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
    return scaler, best_val


def generate_global_noisy_test(test_data, noise_std, scaler, base_seed, n_runs=N_RUNS):
    """Pre-generate N_RUNS global noisy test sets.

    All models share the SAME noisy data for fair comparison.
    Each run uses a different global noise pattern; results are averaged.

    Args:
        test_data: raw test data (T, C)
        noise_std: noise std relative to data std (e.g. 0.03 = 3%)
        scaler: fitted StandardScaler
        base_seed: base random seed
        n_runs: number of different noise patterns

    Returns:
        list of (noisy_X, true_Y) tuples, one per run
    """
    test_n = scaler.transform(test_data)
    test_X, test_Y = make_windows(test_n, SEQ_LEN, PRED_LEN)
    data_std = test_data.std()
    norm_std = test_n.std()

    batches = []
    for i in range(n_runs):
        # Each run uses a deterministic, reproducible noise pattern
        rng = np.random.RandomState(base_seed + i * 1000)
        noise = rng.randn(*test_X.shape).astype(np.float32) * noise_std
        # Scale noise to match normalized magnitude
        noise = noise * (data_std / (norm_std + 1e-8))
        batches.append((test_X + noise, test_Y))
    return batches


@torch.no_grad()
def evaluate(model, noisy_X, true_Y, device='cpu'):
    """Compute MSE on pre-generated noisy test data (global noise)."""
    X_t = torch.FloatTensor(noisy_X).to(device)
    Y_t = torch.FloatTensor(true_Y).to(device)
    pred = model(X_t)
    if pred.shape[1] != PRED_LEN:
        pred = pred[:, -PRED_LEN:, :]
    mse = F.mse_loss(pred, Y_t).item()
    return mse


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Noise: {NOISE_LEVELS}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []

    for ds_name in DATASETS:
        print(f"\n{'='*60}\nDataset: {ds_name}")
        data = load_dataset(ds_name)
        tr, vl, ts = split_data(data)
        n_channels = data.shape[1]
        print(f"  Shape: {data.shape} ch={n_channels}")

        for model_name in MODELS:
            base_cls, pc_cls = MODEL_CLASSES[model_name]
            print(f"\n  --- {model_name} ---")

            # Train Base
            torch.manual_seed(args.seed)
            base_model = base_cls(SEQ_LEN, PRED_LEN, n_channels).to(device)
            base_scaler, _ = train_one(base_model, tr, vl, args.epochs, device=device)

            # Train PC
            torch.manual_seed(args.seed)
            pc_model = pc_cls(SEQ_LEN, PRED_LEN, n_channels).to(device)
            pc_scaler, _ = train_one(pc_model, tr, vl, args.epochs, device=device)

            for sigma in NOISE_LEVELS:
                # Pre-generate global noisy test data (shared by Base & PC)
                global_seed = args.seed + hash(ds_name) % 10000
                noisy_batches_base = generate_global_noisy_test(
                    ts, sigma, base_scaler, global_seed)
                noisy_batches_pc = generate_global_noisy_test(
                    ts, sigma, pc_scaler, global_seed)

                base_mses, pc_mses = [], []
                for run_i in range(N_RUNS):
                    noisy_X_b, true_Y_b = noisy_batches_base[run_i]
                    noisy_X_p, true_Y_p = noisy_batches_pc[run_i]
                    # NOTE: Base and PC use different scalers, so noisy_X differs.
                    # Within each variant, the noise pattern is globally fixed.
                    base_mses.append(evaluate(base_model, noisy_X_b, true_Y_b, device))
                    pc_mses.append(evaluate(pc_model, noisy_X_p, true_Y_p, device))

                b_mse = np.mean(base_mses)
                p_mse = np.mean(pc_mses)
                results.append({'dataset': ds_name, 'model': model_name,
                               'variant': 'base', 'noise_std': sigma, 'mse': round(b_mse, 6)})
                results.append({'dataset': ds_name, 'model': model_name,
                               'variant': 'pc', 'noise_std': sigma, 'mse': round(p_mse, 6)})
                print(f"    sigma={sigma:.2f}: base={b_mse:.4f} pc={p_mse:.4f} "
                      f"delta={(p_mse-b_mse)/b_mse*100:+.1f}%")

    return results


def save_and_report(results):
    df = pd.DataFrame(results)
    agg = df.groupby(['dataset', 'model', 'variant', 'noise_std'])['mse'].mean().reset_index()

    csv_path = os.path.join(OUTPUT_DIR, "noise_results.csv")
    agg.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(agg)} rows)")

    # Summary table (dynamic from data)
    noise_levels = sorted(agg['noise_std'].unique())
    datasets = sorted(agg['dataset'].unique())
    models_list = sorted(agg['model'].unique())

    print(f"\n{'='*70}")
    print("Noise Robustness: Base vs PC MSE")
    for ds_name in datasets:
        print(f"\n  {ds_name}:")
        header_cols = []
        for s in noise_levels:
            header_cols.append(f"Base sigma={s:.2f}")
            header_cols.append(f"PC sigma={s:.2f}")
        print(f"  {'Model':<14} " + " ".join(f"{h:>14}" for h in header_cols))
        for m in models_list:
            row = agg[(agg['dataset'] == ds_name) & (agg['model'] == m)]
            vals = []
            for s in noise_levels:
                for v in ['base', 'pc']:
                    r = row[(row['noise_std'] == s) & (row['variant'] == v)]
                    vals.append(f"{r['mse'].values[0]:.4f}" if len(r) > 0 else "N/A")
            print(f"  {m:<14} " + " ".join(f"{v:>14}" for v in vals))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--models', type=str, default=None)
    args = parser.parse_args()

    global DATASETS, MODELS
    if args.datasets:
        DATASETS = [d.strip() for d in args.datasets.split(',')]
    if args.models:
        MODELS = [m.strip() for m in args.models.split(',')]

    results = run(args)
    save_and_report(results)
    print("\nDone.")


if __name__ == '__main__':
    main()
