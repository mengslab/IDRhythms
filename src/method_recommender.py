
from __future__ import annotations
import numpy as np
import pandas as pd
from src.rhythm_methods import METHOD_ORDER

def _profile(time_points, values):
    t = np.asarray(time_points, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]; y = y[mask]
    if len(t) < 3:
        return {"n_points": int(len(t)), "regular": False, "short": True, "noisy": False, "damped": False, "asymmetric": False, "multifrequency": False}
    order = np.argsort(t); t = t[order]; y = y[order]
    dt = np.diff(t); pos = dt[np.isfinite(dt) & (dt > 0)]
    cv_dt = float(np.std(pos) / max(np.mean(pos), 1e-12)) if len(pos) else np.nan
    regular = bool(np.isfinite(cv_dt) and cv_dt < 0.05)
    short = len(y) < 24
    yc = y - np.mean(y)
    snr_proxy = float(np.std(yc) / max(np.std(np.diff(y)) if len(y) > 1 else 1e-12, 1e-12))
    noisy = snr_proxy < 1.25
    env = np.abs(yc); damped = False
    if len(env) >= 10:
        slope = np.polyfit(np.arange(len(env)), np.log(env + 1e-6), 1)[0]
        damped = bool(abs(slope) > 0.01)
    s = np.std(yc) + 1e-9
    skew = float(np.mean(((y - np.mean(y))/s)**3))
    asymmetric = abs(skew) > 0.5
    power = np.abs(np.fft.rfft(yc))
    multifrequency = bool(np.sum(power > (np.max(power) * 0.45 if len(power) else 1)) >= 2) if len(power) else False
    return {"n_points": int(len(y)), "regular": regular, "short": short, "noisy": noisy, "damped": damped, "asymmetric": asymmetric, "multifrequency": multifrequency}

def recommend_methods(time_points, values):
    p = _profile(time_points, values)
    scores = {m: 0.0 for m in METHOD_ORDER}; why = {m: [] for m in METHOD_ORDER}
    if p["regular"]:
        for m in ["Cosinor","Harmonic regression","FFT","ARSER-like","JTK-like"]:
            scores[m] += 1.2; why[m].append("regular sampling supports this method")
    else:
        scores["Lomb–Scargle"] += 2.0; why["Lomb–Scargle"].append("uneven sampling strongly favors Lomb–Scargle")
        scores["RAIN"] += 1.5; why["RAIN"].append("RAIN can test arbitrary waveforms at prespecified period under non-ideal sampling")
    if p["short"]:
        for m in ["Matrix pencil","Bayesian harmonic","Cosinor"]:
            scores[m] += 1.6
    if p["noisy"]:
        for m in ["Bayesian harmonic","JTK-like","RAIN","Matrix pencil"]:
            scores[m] += 1.2
    if p["damped"]:
        scores["Matrix pencil"] += 2.2; why["Matrix pencil"].append("matrix pencil is appropriate for damped exponentials")
        scores["Wavelet"] += 1.0
    if p["asymmetric"]:
        scores["RAIN"] += 2.0; scores["JTK-like"] += 0.7
    if p["multifrequency"]:
        scores["Harmonic regression"] += 1.6; scores["Matrix pencil"] += 1.4; scores["FFT"] += 0.6
    base = {
        "Matrix pencil": ("estimates damped complex exponentials", 1.0),
        "Cosinor": ("interpretable sinusoidal fit", 1.0),
        "Harmonic regression": ("captures multi-harmonic structure", 1.0),
        "Lomb–Scargle": ("strong default for period discovery", 0.8),
        "FFT": ("fast spectral overview", 0.5),
        "Wavelet": ("time-local rhythm support", 0.8),
        "Autocorrelation": ("coarse rhythm validation", 0.5),
        "JTK-like": ("template-based rank correlation screening", 0.8),
        "ARSER-like": ("AR spectral estimation plus harmonic regression", 0.8),
        "Bayesian harmonic": ("uncertainty-aware harmonic inference", 0.9),
        "RAIN": ("nonparametric umbrella/rise-fall rhythmicity test", 1.0),
    }
    for m,(reason,val) in base.items():
        scores[m] += val; why[m].append(reason)
    rows = [{"method": m, "recommendation_score": round(scores[m], 3), "recommended": scores[m] >= 2.0, "why": "; ".join(why[m])} for m in METHOD_ORDER]
    df = pd.DataFrame(rows).sort_values(["recommendation_score","method"], ascending=[False,True]).reset_index(drop=True)
    profile_bits = []
    profile_bits.append("evenly sampled" if p["regular"] else "unevenly sampled")
    if p["damped"]: profile_bits.append("damped/nonstationary")
    if p["short"]: profile_bits.append("short")
    if p["noisy"]: profile_bits.append("noisy")
    if p["asymmetric"]: profile_bits.append("asymmetric")
    if p["multifrequency"]: profile_bits.append("multi-frequency")
    summary = "Signal profile: " + ", ".join(profile_bits) + ". Top recommendations: " + ", ".join([f"{r.method} ({r.recommendation_score:.2f})" for r in df.head(3).itertuples()]) + "."
    return df, summary, p
