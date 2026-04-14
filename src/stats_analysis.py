
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

def benjamini_hochberg(p_values):
    p = np.asarray([np.nan if pd.isna(x) else float(x) for x in p_values], dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0: return q.tolist()
    pv = p[valid]; order = np.argsort(pv); ranked = pv[order]
    adj = np.empty_like(ranked); prev = 1.0; m = len(ranked)
    for i in range(m - 1, -1, -1):
        prev = min(prev, ranked[i] * m / (i + 1)); adj[i] = min(prev, 1.0)
    out = np.empty_like(pv); out[order] = adj; q[valid] = out
    return q.tolist()

def hedges_g(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    pooled = (((nx - 1) * np.var(x, ddof=1)) + ((ny - 1) * np.var(y, ddof=1))) / max(nx + ny - 2, 1)
    if pooled <= 0: return np.nan
    d = (np.mean(y) - np.mean(x)) / math.sqrt(pooled)
    return d * (1 - (3 / max(4 * (nx + ny) - 9, 1)))

def circular_mean_deg(values_deg):
    v = np.asarray(values_deg, float); v = v[np.isfinite(v)]
    if len(v) == 0: return np.nan
    rad = np.deg2rad(v)
    return math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))

def phase_difference_deg(treated_deg, control_deg):
    tm = circular_mean_deg(treated_deg); cm = circular_mean_deg(control_deg)
    if pd.isna(tm) or pd.isna(cm): return np.nan
    diff = tm - cm
    while diff <= -180: diff += 360
    while diff > 180: diff -= 360
    return diff

def permutation_pvalue(x, y, n_perm=2000, seed=42):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 1 or len(y) < 1: return np.nan
    obs = abs(np.mean(y) - np.mean(x)); pooled = np.concatenate([x, y]).copy(); nx = len(x)
    rng = np.random.default_rng(seed); count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(np.mean(pooled[:nx]) - np.mean(pooled[nx:]))
        if d >= obs - 1e-15: count += 1
    return (count + 1) / (n_perm + 1)

def permutation_phase_pvalue(x_deg, y_deg, n_perm=2000, seed=42):
    x = np.asarray(x_deg, float); y = np.asarray(y_deg, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 1 or len(y) < 1: return np.nan
    obs = abs(phase_difference_deg(y, x)); pooled = np.concatenate([x, y]).copy(); nx = len(x)
    rng = np.random.default_rng(seed); count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(phase_difference_deg(pooled[nx:], pooled[:nx]))
        if d >= obs - 1e-15: count += 1
    return (count + 1) / (n_perm + 1)

def compare_groups(replicate_params, parameter, test="permutation", phase_parameter=False, n_perm=2000):
    out = []
    for feature, df in replicate_params.groupby("feature", sort=False):
        control = df[df["condition"] == "control"][parameter].to_numpy(float)
        treated = df[df["condition"] == "treated"][parameter].to_numpy(float)
        nc = np.isfinite(control).sum(); nt = np.isfinite(treated).sum()
        cm = circular_mean_deg(control) if phase_parameter else np.nanmean(control) if nc else np.nan
        tm = circular_mean_deg(treated) if phase_parameter else np.nanmean(treated) if nt else np.nan
        effect = phase_difference_deg(treated, control) if phase_parameter else (tm - cm if nc and nt else np.nan)
        if phase_parameter:
            p = permutation_phase_pvalue(control, treated, n_perm=n_perm); eg = np.nan
        else:
            if test == "welch": p = float(scipy_stats.ttest_ind(control, treated, equal_var=False, nan_policy="omit").pvalue)
            elif test == "mannwhitney": p = float(scipy_stats.mannwhitneyu(control, treated, alternative="two-sided").pvalue)
            else: p = permutation_pvalue(control, treated, n_perm=n_perm)
            eg = hedges_g(control, treated)
        out.append({"feature": feature, "parameter": parameter, "n_control": int(nc), "n_treated": int(nt), "control_mean": cm, "treated_mean": tm, "effect": effect, "effect_size_hedges_g": eg, "p_value": p})
    res = pd.DataFrame(out)
    if not res.empty: res["q_value"] = benjamini_hochberg(res["p_value"].tolist())
    return res

def build_stats_summary(results_df, parameter):
    if results_df is None or results_df.empty: return f"No valid control-versus-treated comparison could be computed for {parameter}."
    sig = results_df[results_df["q_value"].fillna(1) < 0.05]
    if sig.empty: return f"No features reached FDR < 0.05 for {parameter} in the current comparison."
    top = sig.sort_values(["q_value", "p_value"]).head(5)
    phrases = [f"{row['feature']} (control {row['control_mean']:.3g}, treated {row['treated_mean']:.3g}, effect {row['effect']:.3g}, q={row['q_value']:.3g})" for _, row in top.iterrows()]
    return f"Significant control-versus-treated differences for {parameter} were detected in {len(sig)} feature(s), including " + "; ".join(phrases) + "."
