"""
Predict-then-Correct: Simplified correction operators.

Two correction modes with rigorous mathematical foundations:

  - linear_correction  - OLS ratio in time domain (Gauss-Markov theorem)
  - spectral_correction - Auto-correlation ratio via Wiener-Khinchin theorem

Both return y_corrected = alpha * y_pred, shape (B, pred_len, C).
alpha is per-channel: (B, 1, C).
"""

import torch
import torch.fft


def _linear_alpha(x_true, x_recon, eps=1e-6):
    num = (x_true * x_recon).sum(dim=1, keepdim=True)
    den = (x_recon * x_recon).sum(dim=1, keepdim=True) + eps
    return num / den


def _spectral_alpha(x_true, x_recon, eps=1e-6):
    x_true_float = x_true.float()
    x_recon_float = x_recon.float()

    X_true = torch.fft.rfft(x_true_float, dim=1, norm='ortho')
    X_recon = torch.fft.rfft(x_recon_float, dim=1, norm='ortho')

    S_true = (X_true * X_true.conj()).real
    S_recon = (X_recon * X_recon.conj()).real

    R_true = torch.fft.irfft(S_true, n=x_true.shape[1], dim=1, norm='ortho').clamp(min=0)
    R_recon = torch.fft.irfft(S_recon, n=x_recon.shape[1], dim=1, norm='ortho').clamp(min=0)

    num = (R_true * R_recon).sum(dim=1, keepdim=True)
    den = (R_recon * R_recon).sum(dim=1, keepdim=True) + eps
    return (num / den).sqrt().to(dtype=x_recon.dtype)


def _fused_alpha(x_true, x_recon, delta=0.15, lam=0.3, eps=1e-6):
    alpha_lin = _linear_alpha(x_true, x_recon, eps=eps)
    alpha_spec = _spectral_alpha(x_true, x_recon, eps=eps)

    diff_rel = (alpha_lin - alpha_spec).abs() / (
        torch.max(alpha_lin.abs(), alpha_spec.abs()) + eps
    )
    agree = (diff_rel < delta).float()

    alpha_avg = (alpha_lin + alpha_spec) / 2
    alpha_shrunk = 1.0 + lam * (alpha_avg - 1.0)
    alpha_final = agree * alpha_avg + (1 - agree) * alpha_shrunk
    return {
        'alpha': alpha_final,
        'alpha_lin': alpha_lin,
        'alpha_spec': alpha_spec,
        'agree': agree,
    }


def _multiband_alpha(x_true, x_recon, n_bands=3, eps=1e-6):
    Xt = torch.fft.rfft(x_true.float(), dim=1, norm='ortho')
    Xr = torch.fft.rfft(x_recon.float(), dim=1, norm='ortho')

    St = (Xt * Xt.conj()).real
    Sr = (Xr * Xr.conj()).real

    nf = St.shape[1]
    band_edges = torch.linspace(0, nf, n_bands + 1, device=St.device).long()

    alphas = []
    energies = []
    for k in range(n_bands):
        start = band_edges[k].item()
        end = band_edges[k + 1].item()
        Et = St[:, start:end, :].sum(dim=1, keepdim=True) + eps
        Er = Sr[:, start:end, :].sum(dim=1, keepdim=True) + eps
        alphas.append((Et / Er).sqrt())
        energies.append(Et)

    alphas = torch.stack(alphas, dim=-1)
    energies = torch.stack(energies, dim=-1)
    weights = energies / (energies.sum(dim=-1, keepdim=True) + eps)
    return (alphas * weights).sum(dim=-1).to(dtype=x_recon.dtype)


def compute_correction_alpha(
    x_true,
    x_recon,
    correction_type='linear',
    return_components=False,
    **kwargs,
):
    """
    Compute per-sample, per-channel correction factors.

    Returns an alpha tensor with shape (B, 1, C), or a component dict when
    return_components=True. This exposes the same correction factors used by
    apply_correction without changing the public prediction behavior.
    """
    import logging
    logger = logging.getLogger(__name__)

    t = correction_type.lower() if correction_type else 'none'
    eps = kwargs.get('eps', 1e-6)

    if t in ('linear', 'b'):
        alpha = _linear_alpha(x_true, x_recon, eps=eps)
        components = {'alpha': alpha, 'alpha_lin': alpha}
    elif t in ('s', 'spectral'):
        alpha = _spectral_alpha(x_true, x_recon, eps=eps)
        components = {'alpha': alpha, 'alpha_spec': alpha}
    elif t == 'fused':
        components = _fused_alpha(
            x_true,
            x_recon,
            delta=kwargs.get('delta', 0.15),
            lam=kwargs.get('lam', 0.3),
            eps=eps,
        )
    elif t == 'gated':
        alpha_raw = _linear_alpha(x_true, x_recon, eps=eps)
        x_recon_corrected = alpha_raw * x_recon
        err = torch.norm(x_true - x_recon_corrected, dim=1, keepdim=True) / (
            torch.norm(x_true, dim=1, keepdim=True) + eps
        )
        tau = kwargs.get('tau', 0.1)
        confidence = torch.exp(-err ** 2 / tau ** 2)
        alpha = 1.0 + confidence * (alpha_raw - 1.0)
        components = {'alpha': alpha, 'alpha_lin': alpha_raw, 'confidence': confidence}
    elif t == 'multiband':
        alpha = _multiband_alpha(
            x_true,
            x_recon,
            n_bands=kwargs.get('n_bands', 3),
            eps=eps,
        )
        components = {'alpha': alpha}
    elif t == 'none':
        alpha = torch.ones(
            x_true.shape[0],
            1,
            x_true.shape[2],
            dtype=x_true.dtype,
            device=x_true.device,
        )
        components = {'alpha': alpha}
    else:
        logger.warning(
            "Unknown correction_type '%s', falling back to identity.", correction_type
        )
        alpha = torch.ones(
            x_true.shape[0],
            1,
            x_true.shape[2],
            dtype=x_true.dtype,
            device=x_true.device,
        )
        components = {'alpha': alpha}

    return components if return_components else components['alpha']


# ============================================================================
# linear_correction - OLS ratio (Gauss-Markov theorem)
#
#   argmin_alpha ||x_true - alpha * x_recon||^2
#   => alpha = <x_true, x_recon> / ||x_recon||^2   (per-channel)
#
# This is the BLUE (Best Linear Unbiased Estimator) under the
# Gauss-Markov assumptions. It corrects systematic amplitude bias
# in the time domain - no FFT needed.
# ============================================================================

def linear_correction(x_true, x_recon, y_pred, eps=1e-6):
    """
    OLS ratio correction - Gauss-Markov theorem.

    Computes the optimal linear scaling factor per channel:
        alpha_c = <x_true_c, x_recon_c> / ||x_recon_c||^2

    Args:
        x_true:  ground truth history  (B, L, C)
        x_recon: reconstructed history (B, L, C)
        y_pred:  raw prediction        (B, P, C)
        eps:     numerical stability

    Returns:
        y_corrected = alpha * y_pred   (B, P, C)
    """
    num = (x_true * x_recon).sum(dim=1, keepdim=True)         # (B, 1, C)
    den = (x_recon * x_recon).sum(dim=1, keepdim=True) + eps
    alpha = num / den
    return alpha * y_pred


# ============================================================================
# spectral_correction - Auto-correlation ratio (Wiener-Khinchin theorem)
#
#   Wiener-Khinchin: S(f) = |X(f)|^2 = F{R(tau)}
#   where R(tau) = E[x(t)·x(t+tau)] is the auto-correlation function.
#
#   Steps:
#   1. FFT -> power spectrum S(f) = |X(f)|^2
#   2. IFFT -> auto-correlation R(tau)
#   3. Per-channel ratio: alpha_c^2 = <R_true, R_recon> / ||R_recon||^2
#
#   This corrects temporal-structure bias (e.g. if the model
#   systematically dampens certain frequency components).
# ============================================================================

def spectral_correction(x_true, x_recon, y_pred, eps=1e-6):
    """
    Power-spectrum correction - Wiener-Khinchin theorem.

    Compares auto-correlation functions R(tau) of true vs. reconstructed
    history, derived from the power spectrum via FFT/IFFT.

    Args:
        x_true:  ground truth history  (B, L, C)
        x_recon: reconstructed history (B, L, C)
        y_pred:  raw prediction        (B, P, C)
        eps:     numerical stability

    Returns:
        y_corrected = alpha * y_pred   (B, P, C)
    """
    # FFT along time axis
    X_true = torch.fft.rfft(x_true.float(), dim=1, norm='ortho')
    X_recon = torch.fft.rfft(x_recon.float(), dim=1, norm='ortho')

    # Power spectrum: S(f) = |X(f)|^2   (Wiener-Khinchin step 1)
    S_true = (X_true * X_true.conj()).real
    S_recon = (X_recon * X_recon.conj()).real

    # Auto-correlation: R(tau) = F^{-1}{S(f)}   (Wiener-Khinchin step 2)
    R_true = torch.fft.irfft(S_true, dim=1, norm='ortho').clamp(min=0)
    R_recon = torch.fft.irfft(S_recon, dim=1, norm='ortho').clamp(min=0)

    # Per-channel least-squares ratio on auto-correlations
    num = (R_true * R_recon).sum(dim=1, keepdim=True)          # (B, 1, C)
    den = (R_recon * R_recon).sum(dim=1, keepdim=True) + eps
    alpha = (num / den).sqrt()
    return alpha * y_pred


# ============================================================================
# fused_correction - Disagreement-gated fusion of linear + spectral
#
#   Computes alpha_lin (OLS) and alpha_spec (Wiener-Khinchin) independently.
#   When they agree -> trust the average.
#   When they disagree -> shrink toward 1 (conservative).
#
#   Key insight: disagreement between two independent estimators signals
#   uncertainty - no extra model needed.
# ============================================================================

def fused_correction(x_true, x_recon, y_pred, delta=0.15, lam=0.3, eps=1e-6):
    """
    Disagreement-gated fusion of linear and spectral correction.

    Two independent estimators (time-domain OLS, frequency-domain
    autocorrelation) each produce a per-channel alpha. If they agree
    (relative difference < delta), use their average. If they disagree,
    the correction is uncertain - shrink toward identity.

    Args:
        x_true:  ground truth history  (B, L, C)
        x_recon: reconstructed history (B, L, C)
        y_pred:  raw prediction        (B, P, C)
        delta:   disagreement threshold (relative)
        lam:     shrinkage factor when disagreeing (0=identity, 1=full)
        eps:     numerical stability

    Returns:
        y_corrected  (B, P, C)
    """
    # --- Linear (time domain) ---
    num_lin = (x_true * x_recon).sum(dim=1, keepdim=True)        # (B, 1, C)
    den_lin = (x_recon * x_recon).sum(dim=1, keepdim=True) + eps
    alpha_lin = num_lin / den_lin

    # --- Spectral (frequency domain) ---
    Xt = torch.fft.rfft(x_true.float(), dim=1, norm='ortho')
    Xr = torch.fft.rfft(x_recon.float(), dim=1, norm='ortho')
    St = (Xt * Xt.conj()).real
    Sr = (Xr * Xr.conj()).real
    Rt = torch.fft.irfft(St, dim=1, norm='ortho').clamp(min=0)
    Rr = torch.fft.irfft(Sr, dim=1, norm='ortho').clamp(min=0)
    num_spec = (Rt * Rr).sum(dim=1, keepdim=True)
    den_spec = (Rr * Rr).sum(dim=1, keepdim=True) + eps
    alpha_spec = (num_spec / den_spec).sqrt()

    # --- Disagreement gate ---
    diff_rel = (alpha_lin - alpha_spec).abs() / (
        torch.max(alpha_lin.abs(), alpha_spec.abs()) + eps)       # (B, 1, C)
    agree = (diff_rel < delta).float()                            # 1 if agree

    alpha_avg = (alpha_lin + alpha_spec) / 2
    alpha_shrunk = 1.0 + lam * (alpha_avg - 1.0)                 # shrink toward 1

    alpha_final = agree * alpha_avg + (1 - agree) * alpha_shrunk
    return alpha_final * y_pred


# ============================================================================
# multiband_correction - Per-frequency-band spectral correction
#
#   Instead of a single alpha per channel (comparing full-spectrum
#   autocorrelations), split the frequency axis into K bands, compute
#   per-band energy ratios independently, then aggregate weighted by
#   each band's energy contribution.
#
#   This captures frequency-selective bias: e.g. the model might
#   attenuate high frequencies more than low frequencies.
# ============================================================================

def multiband_correction(x_true, x_recon, y_pred, n_bands=3, eps=1e-6):
    """
    Multi-band spectral correction.

    Splits the frequency range into n_bands, computes sqrt(E_true/E_recon)
    per band per channel, then aggregates by energy-weighted average.

    Args:
        x_true:  ground truth history  (B, L, C)
        x_recon: reconstructed history (B, L, C)
        y_pred:  raw prediction        (B, P, C)
        n_bands: number of frequency bands
        eps:     numerical stability

    Returns:
        y_corrected  (B, P, C)
    """
    Xt = torch.fft.rfft(x_true.float(), dim=1, norm='ortho')     # (B, nf, C)
    Xr = torch.fft.rfft(x_recon.float(), dim=1, norm='ortho')

    # Power spectra
    St = (Xt * Xt.conj()).real                                    # (B, nf, C)
    Sr = (Xr * Xr.conj()).real

    nf = St.shape[1]
    band_edges = torch.linspace(0, nf, n_bands + 1, device=St.device).long()

    alphas = []
    energies = []

    for k in range(n_bands):
        start = band_edges[k].item()
        end = band_edges[k + 1].item()

        # Per-band energy: sum of power spectrum in band
        Et = St[:, start:end, :].sum(dim=1, keepdim=True) + eps  # (B, 1, C)
        Er = Sr[:, start:end, :].sum(dim=1, keepdim=True) + eps

        # Per-band correction: sqrt(energy ratio)
        alpha_k = (Et / Er).sqrt()
        alphas.append(alpha_k)
        energies.append(Et)                                       # weight by true energy

    # Weighted average across bands
    alphas = torch.stack(alphas, dim=-1)                          # (B, 1, C, K)
    energies = torch.stack(energies, dim=-1)
    weights = energies / (energies.sum(dim=-1, keepdim=True) + eps)

    alpha_final = (alphas * weights).sum(dim=-1)                  # (B, 1, C)
    return alpha_final * y_pred


# ============================================================================
# gated_correction - Confidence-gated linear correction
#
#   Standard approach: blindly apply alpha regardless of reconstruction quality.
#   Our approach: gate the correction by how well the model reconstructs the past.
#
#   alpha_raw = OLS(x, x_hat)                    - raw correction
#   err = ||x - alpha_raw * x_hat|| / ||x||         - relative reconstruction error
#   gamma = exp(-err^2 / tau^2)                        - confidence gate
#   alpha_final = 1 + gamma * (alpha_raw - 1)    - shrink to 1 when uncertain
#
#   Training dynamics: early epochs -> poor recon -> gamma ~ 0 -> model learns from
#   raw loss. Later epochs -> good recon -> gamma ~ 1 -> correction fully engaged.
# ============================================================================

def gated_correction(x_true, x_recon, y_pred, tau=0.1, eps=1e-6):
    """
    Confidence-gated linear correction.

    Raw OLS correction is dampened toward identity (alpha=1) when the
    corrected reconstruction still has large error. This prevents
    unreliable corrections from amplifying errors.

    Args:
        x_true:  ground truth history  (B, L, C)
        x_recon: reconstructed history (B, L, C)
        y_pred:  raw prediction        (B, P, C)
        tau:     temperature for confidence decay (smaller = stricter gate)
        eps:     numerical stability

    Returns:
        y_corrected  (B, P, C)
    """
    # Raw OLS correction
    num = (x_true * x_recon).sum(dim=1, keepdim=True)            # (B, 1, C)
    den = (x_recon * x_recon).sum(dim=1, keepdim=True) + eps
    alpha_raw = num / den

    # Corrected reconstruction
    x_recon_corrected = alpha_raw * x_recon

    # Relative reconstruction error per channel
    err = torch.norm(x_true - x_recon_corrected, dim=1, keepdim=True) / (
        torch.norm(x_true, dim=1, keepdim=True) + eps)            # (B, 1, C)

    # Confidence gate: exp(-err^2 / tau^2) in (0, 1]
    confidence = torch.exp(-err ** 2 / tau ** 2)

    # Shrink toward 1 when uncertain
    alpha_final = 1.0 + confidence * (alpha_raw - 1.0)
    return alpha_final * y_pred


# ============================================================================
# Backward-compatible aliases and routing
# ============================================================================

# Maintain compatibility with old code that imports these names
S_Correction_temporal = spectral_correction
B_Correction_temporal = linear_correction  # B_Correction removed; fallback to linear


def apply_correction(x_true, x_recon, y_pred, correction_type='linear', **kwargs):
    """
    Routing function: pick correction by name.

    Args:
        x_true:          ground truth history  (B, L, C)
        x_recon:         reconstructed history (B, L, C)
        y_pred:          raw prediction        (B, P, C)
        correction_type: 'linear' | 'spectral' | 'fused' | 'multiband' | 'none'
                         (legacy 'S'->spectral, 'B'->linear with warning)
        **kwargs:        passed to correction function (e.g. delta, lam, n_bands)

    Returns:
        corrected prediction (B, P, C)
    """
    import logging
    logger = logging.getLogger(__name__)

    t = correction_type.lower() if correction_type else 'none'

    if t in ('linear',):
        return linear_correction(x_true, x_recon, y_pred)
    elif t in ('s', 'spectral'):
        return spectral_correction(x_true, x_recon, y_pred)
    elif t == 'gated':
        tau = kwargs.get('tau', 0.1)
        return gated_correction(x_true, x_recon, y_pred, tau=tau)
    elif t == 'fused':
        delta = kwargs.get('delta', 0.15)
        lam = kwargs.get('lam', 0.3)
        return fused_correction(x_true, x_recon, y_pred, delta=delta, lam=lam)
    elif t == 'multiband':
        n_bands = kwargs.get('n_bands', 3)
        return multiband_correction(x_true, x_recon, y_pred, n_bands=n_bands)
    elif t == 'b':
        logger.warning(
            "correction_type='B' (bispectrum) has been removed - "
            "falling back to 'linear' (OLS ratio, Gauss-Markov)."
        )
        return linear_correction(x_true, x_recon, y_pred)
    elif t == 'none':
        return y_pred
    else:
        logger.warning(
            "Unknown correction_type '%s', falling back to identity.", correction_type
        )
        return y_pred
