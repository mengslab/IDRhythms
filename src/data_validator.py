
from __future__ import annotations
import numpy as np
import pandas as pd

def find_time_column(df: pd.DataFrame):
    for c in df.columns:
        if str(c).strip().lower() == "time":
            return c
    return None

def prepare_time_series_dataframe(df: pd.DataFrame, interval: float = 1.0):
    if df is None or df.empty:
        return {"clean_df": pd.DataFrame(), "time_col": None, "signal_cols": [], "warnings": ["Dataset is empty."], "issues": ["Dataset is empty."], "stats": {"n_rows": 0, "n_signals": 0, "missing_cells": 0, "median_dt": np.nan, "even_sampling": False, "cv_dt": np.nan}}
    out = df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    warnings, issues = [], []
    time_col = find_time_column(out)
    if time_col is None:
        out.insert(0, "time", np.arange(len(out), dtype=float) * float(interval))
        time_col = "time"
        warnings.append("No time column detected. Generated synthetic time axis from sampling interval.")
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce").interpolate(limit_direction="both")
    signal_cols = [c for c in out.columns if c != time_col]
    for c in signal_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    missing_cells = int(out[signal_cols].isna().sum().sum()) if signal_cols else 0
    if missing_cells > 0:
        warnings.append(f"Missing signal values detected ({missing_cells}). Applied interpolation and edge fill.")
        out[signal_cols] = out[signal_cols].interpolate(limit_direction="both")
    out = out.sort_values(time_col).drop_duplicates(subset=[time_col]).reset_index(drop=True)
    t = out[time_col].to_numpy(dtype=float)
    dt = np.diff(t)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(pos)) if len(pos) else np.nan
    cv_dt = float(np.std(pos) / max(np.mean(pos), 1e-12)) if len(pos) else np.nan
    even = bool(np.isfinite(cv_dt) and cv_dt < 0.05)
    if len(out) < 12:
        warnings.append("Short series detected (n < 12). Frequency resolution may be limited.")
    if signal_cols and all(out[c].notna().sum() < 4 for c in signal_cols):
        issues.append("Too few non-missing observations in signal columns.")
    if even: warnings.append("Sampling appears regular.")
    else: warnings.append("Sampling is irregular or mixed.")
    return {
        "clean_df": out, "time_col": time_col, "signal_cols": signal_cols,
        "warnings": warnings, "issues": issues,
        "stats": {"n_rows": int(len(out)), "n_signals": int(len(signal_cols)), "missing_cells": int(missing_cells), "median_dt": median_dt, "even_sampling": even, "cv_dt": cv_dt},
    }

def validation_summary_text(prep: dict) -> str:
    if prep["issues"]:
        return "Validation found blocking issues: " + "; ".join(prep["issues"])
    s = prep["stats"]
    bits = [f"{s['n_rows']} rows", f"{s['n_signals']} signals"]
    if np.isfinite(s["median_dt"]): bits.append(f"median Δt {s['median_dt']:.3g}")
    bits.append("regular sampling" if s["even_sampling"] else "irregular / mixed sampling")
    return "Validation complete: " + ", ".join(bits) + "."
