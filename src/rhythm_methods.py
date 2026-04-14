
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import signal, stats, linalg
import pywt

METHOD_ORDER = [
    "Matrix pencil",
    "Cosinor",
    "Harmonic regression",
    "Lomb–Scargle",
    "FFT",
    "Wavelet",
    "Autocorrelation",
    "JTK-like",
    "ARSER-like",
    "Bayesian harmonic",
    "RAIN",
]

def _as_arrays(time_points, values):
    t = np.asarray(time_points, dtype=float); y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]; y = y[mask]
    order = np.argsort(t)
    return t[order], y[order]

def _median_dt(t):
    if len(t) < 2: return 1.0
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 1.0

def _candidate_periods(t, n_grid=220):
    dt = _median_dt(t); total = max(float(t[-1] - t[0]), dt * 8)
    pmin = max(2.0 * dt, 2.0); pmax = max(min(48.0, total * 0.95), pmin + dt * 2)
    return np.linspace(pmin, pmax, n_grid)

def _ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    rss = float(np.sum((y - fitted) ** 2))
    n = len(y); k = X.shape[1]
    aic = n * math.log(max(rss / max(n, 1), 1e-12)) + 2 * k
    r2 = 1 - rss / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
    return beta, fitted, rss, aic, r2

def _amp_phase(c, s):
    amp = float(np.sqrt(c**2 + s**2))
    phase = float(np.degrees(np.arctan2(s, c)))
    return amp, phase

def _fit_single_cosine(t, y, period):
    X = np.column_stack([np.ones_like(t), np.cos(2*np.pi*t/period), np.sin(2*np.pi*t/period)])
    beta, fitted, rss, aic, r2 = _ols_fit(X, y)
    amp, phase = _amp_phase(beta[1], beta[2])
    return {"period_h": float(period), "amplitude": amp, "phase_deg": phase, "score": float(r2), "p_value": np.nan, "fitted": fitted, "aic": aic, "notes": "Single-frequency fit"}

# Faithful-ish matrix pencil implementation for damped sinusoids
def matrix_pencil_method(t, y):
    dt = _median_dt(t)
    tu = np.arange(t[0], t[-1] + dt * 0.5, dt)
    yu = np.interp(tu, t, y)
    x = yu.astype(complex)
    N = len(x)
    if N < 8:
        return {"method":"Matrix pencil","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"Too few points for matrix pencil","fitted":np.full_like(y, np.nan)}
    L = N // 2
    K = N - L
    Y0 = np.column_stack([x[i:i+L] for i in range(K)])
    Y1 = np.column_stack([x[i+1:i+1+L] for i in range(K-1)])
    Y0s = Y0[:, :K-1]
    U, s, Vh = np.linalg.svd(Y0s, full_matrices=False)
    thresh = s[0] * 1e-3 if len(s) else 0.0
    r = int(max(1, min(np.sum(s > thresh), min(6, len(s)))))
    Ur = U[:, :r]
    Sr = np.diag(s[:r])
    Vr = Vh.conj().T[:, :r]
    A = np.linalg.pinv(Sr) @ Ur.conj().T @ Y1 @ Vr
    eigvals = np.linalg.eigvals(A)
    valid = []
    for lam in eigvals:
        if abs(lam) <= 1e-8: 
            continue
        freq = abs(np.angle(lam)) / (2*np.pi*dt)
        if freq <= 0:
            continue
        period = 1.0 / freq
        if 2 <= period <= max(48, tu[-1]-tu[0]):
            valid.append((lam, period))
    if not valid:
        return {"method":"Matrix pencil","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"No valid oscillatory poles detected","fitted":np.full_like(y, np.nan)}
    # solve amplitudes for candidate poles
    poles = np.array([v[0] for v in valid], dtype=complex)
    n = np.arange(N)
    V = np.column_stack([poles_i**n for poles_i in poles])
    coeffs, *_ = np.linalg.lstsq(V, x, rcond=None)
    amps = np.abs(coeffs)
    idx = int(np.argmax(amps))
    lam = poles[idx]; period = float(valid[idx][1])
    fitted_u = np.real(V @ coeffs)
    fitted = np.interp(t, tu, fitted_u)
    # phase from least-squares cosine fit at selected period for interpretability
    fit = _fit_single_cosine(t, y, period)
    score = float(amps[idx] / max(np.std(y), 1e-9))
    return {"method":"Matrix pencil","period_h":period,"amplitude":float(amps[idx]),"phase_deg":fit["phase_deg"],"p_value":np.nan,"score":score,"notes":f"Matrix pencil with model order {r}","fitted":fitted}

def cosinor_method(t, y):
    best = None
    for p in _candidate_periods(t):
        fit = _fit_single_cosine(t, y, p); fit["notes"] = "Interpretable sinusoidal fit"
        if best is None or fit["aic"] < best["aic"]: best = fit
    return {"method":"Cosinor", **{k:v for k,v in best.items() if k != "aic"}}

def harmonic_regression_method(t, y, n_harmonics=3):
    best = None
    for p in _candidate_periods(t):
        cols = [np.ones_like(t)]
        for h in range(1, n_harmonics+1):
            cols += [np.cos(2*np.pi*h*t/p), np.sin(2*np.pi*h*t/p)]
        X = np.column_stack(cols)
        beta, fitted, rss, aic, r2 = _ols_fit(X, y)
        if best is None or aic < best["aic"]:
            best = {"period": p, "beta": beta, "fitted": fitted, "aic": aic, "r2": r2}
    amps = []
    for h in range(1, n_harmonics+1):
        amp, phase = _amp_phase(best["beta"][1+2*(h-1)], best["beta"][2+2*(h-1)])
        amps.append((amp, phase, h))
    amp, phase, hstar = max(amps, key=lambda x: x[0])
    return {"method":"Harmonic regression", "period_h": float(best["period"]/hstar), "amplitude": float(amp), "phase_deg": float(phase), "p_value": np.nan, "score": float(best["r2"]), "notes": f"{n_harmonics} harmonics; dominant harmonic {hstar}", "fitted": best["fitted"]}

def lomb_scargle_method(t, y):
    periods = _candidate_periods(t)
    y0 = y - np.mean(y); ang = 2*np.pi/periods
    power = signal.lombscargle(t, y0, ang, normalize=True)
    p = float(periods[int(np.argmax(power))]); fit = _fit_single_cosine(t, y, p)
    fit.update({"method": "Lomb–Scargle", "score": float(np.max(power)), "notes": "Normalized periodogram peak"}); fit.pop("aic", None)
    return fit

def fft_method(t, y):
    dt = _median_dt(t); tu = np.arange(t[0], t[-1] + dt*0.5, dt); yu = np.interp(tu, t, y)
    freqs = np.fft.rfftfreq(len(yu), d=dt); spec = np.fft.rfft(yu - np.mean(yu))
    if len(freqs) <= 1:
        return {"method":"FFT","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"Too few points","fitted":np.full_like(y, np.nan)}
    idx = int(np.argmax(np.abs(spec[1:])) + 1); period = float(np.inf if freqs[idx] <= 0 else 1/freqs[idx])
    fit = _fit_single_cosine(t, y, period); fit.update({"method":"FFT","score":float(np.abs(spec[idx])),"notes":"Dominant discrete frequency"}); fit.pop("aic", None)
    return fit

def wavelet_method(t, y):
    dt = _median_dt(t); periods = _candidate_periods(t)
    scales = periods / (dt * pywt.central_frequency("morl"))
    coef, _ = pywt.cwt(y - np.mean(y), scales, "morl", sampling_period=dt)
    power = np.abs(coef)**2; idx = int(np.argmax(power.mean(axis=1)))
    fit = _fit_single_cosine(t, y, float(periods[idx])); fit.update({"method":"Wavelet","score":float(power.mean(axis=1)[idx]),"notes":"Global Morlet wavelet power"}); fit.pop("aic", None)
    return fit

def autocorrelation_method(t, y):
    dt = _median_dt(t); y0 = y - np.mean(y)
    acf = signal.correlate(y0, y0, mode="full"); acf = acf[len(acf)//2:]; acf = acf / max(acf[0], 1e-12)
    peaks, _ = signal.find_peaks(acf[1:], distance=max(1, int(2/dt)))
    if len(peaks) == 0:
        return {"method":"Autocorrelation","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"No ACF peak","fitted":np.full_like(y, np.nan)}
    lag = int(peaks[np.argmax(acf[1:][peaks])] + 1)
    fit = _fit_single_cosine(t, y, float(lag*dt)); fit.update({"method":"Autocorrelation","score":float(acf[lag]),"notes":"Strongest autocorrelation peak"}); fit.pop("aic", None)
    return fit

# JTK-like faithful-in-spirit: scan period/phase templates using rank-based Kendall tau
def jtk_like_method(t, y):
    periods = _candidate_periods(t)
    best = None
    for p in periods:
        phases = np.linspace(0, p, 24, endpoint=False)
        for phase in phases:
            template = np.cos(2*np.pi*(t-phase)/p)
            tau, pval = stats.kendalltau(y, template)
            score = abs(float(tau)) if np.isfinite(tau) else -np.inf
            if best is None or (np.isfinite(pval) and pval < best["pval"]) or (not np.isfinite(best["pval"]) and score > best["score"]):
                best = {"period": float(p), "phase_shift": float(phase), "score": score, "pval": float(pval) if np.isfinite(pval) else np.nan}
    fit = _fit_single_cosine(t, y, best["period"]); fit.update({"method":"JTK-like","p_value":best["pval"],"score":best["score"],"notes":"Kendall-tau template scan over period/phase grid"}); fit.pop("aic", None)
    return fit

# ARSER-like: autoregressive spectral estimation + harmonic regression
def _yw_ar(x, order):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    r = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(order+1)])
    R = linalg.toeplitz(r[:-1])
    try:
        a = np.linalg.solve(R + 1e-8*np.eye(order), r[1:])
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R + 1e-8*np.eye(order), r[1:], rcond=None)[0]
    sigma2 = max(r[0] - np.dot(a, r[1:]), 1e-9)
    return a, sigma2

def arser_like_method(t, y):
    dt = _median_dt(t)
    tu = np.arange(t[0], t[-1] + dt*0.5, dt)
    yu = np.interp(tu, t, y)
    yc = yu - np.mean(yu)
    n = len(yc)
    if n < 8:
        return {"method":"ARSER-like","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"Too few points","fitted":np.full_like(y, np.nan)}
    max_order = max(2, min(12, n//3))
    best = None
    for order in range(2, max_order+1):
        a, sigma2 = _yw_ar(yc, order)
        aic = n * np.log(sigma2) + 2 * order
        if best is None or aic < best["aic"]:
            best = {"order": order, "a": a, "sigma2": sigma2, "aic": aic}
    a = best["a"]
    freqs = np.linspace(1e-4, 0.5/dt, 2048)
    omega = 2*np.pi*freqs*dt
    denom = np.abs(1 - np.sum([a[k]*np.exp(-1j*omega*(k+1)) for k in range(len(a))], axis=0))**2
    psd = best["sigma2"] / np.maximum(denom, 1e-12)
    idx = int(np.argmax(psd))
    period = float(1/freqs[idx]) if freqs[idx] > 0 else np.nan
    fit = _fit_single_cosine(t, y, period)
    fit.update({"method":"ARSER-like","score":float(np.max(psd)),"notes":f"Yule-Walker AR spectral estimate, order {best['order']}"})
    fit.pop("aic", None)
    return fit

def bayesian_rhythmic_method(t, y):
    best = None; alpha = 1.0
    for p in _candidate_periods(t):
        X = np.column_stack([np.ones_like(t), np.cos(2*np.pi*t/p), np.sin(2*np.pi*t/p)])
        beta, fitted, rss, aic, r2 = _ols_fit(X, y)
        n, m = X.shape; sigma2 = max(rss / max(n - m, 1), 1e-6)
        S0_inv = alpha * np.eye(m); SN_inv = S0_inv + (1.0/sigma2) * X.T @ X
        SN = np.linalg.inv(SN_inv); mN = (1.0/sigma2) * SN @ X.T @ y
        logev = -0.5*(np.linalg.slogdet(SN_inv)[1] + n*np.log(2*np.pi*sigma2))
        if best is None or logev > best["logev"]:
            best = {"period": p, "mN": mN, "SN": SN, "logev": float(logev), "X": X}
    rng = np.random.default_rng(42); samples = rng.multivariate_normal(best["mN"], best["SN"], size=1200)
    amp_samples = np.sqrt(samples[:,1]**2 + samples[:,2]**2); amp = float(np.mean(amp_samples))
    phase = float(np.degrees(np.arctan2(best["mN"][2], best["mN"][1]))); lo, hi = np.quantile(amp_samples, [0.025, 0.975])
    return {"method":"Bayesian harmonic","period_h":float(best["period"]),"amplitude":amp,"phase_deg":phase,"p_value":np.nan,"score":float(best["logev"]),"notes":f"95% CrI amplitude [{lo:.3g}, {hi:.3g}]","fitted":best["X"] @ best["mN"]}

# RAIN faithful-in-spirit: prespecified-period rise/fall umbrella scan with rank-based p-values
def rain_method(t, y):
    periods = _candidate_periods(t)
    best = None
    for p in periods:
        phase_offsets = np.linspace(0, p, 24, endpoint=False)
        for phase in phase_offsets:
            ph = ((t - phase) % p) / p
            order = np.argsort(ph)
            yy = y[order]; phs = ph[order]
            n = len(yy)
            for peak_idx in range(max(1, n//4), min(n-1, 3*n//4)):
                rise = yy[:peak_idx+1]
                fall = yy[peak_idx:]
                # monotone increasing then decreasing rank correlations
                tau1, p1 = stats.kendalltau(np.arange(len(rise)), rise)
                tau2, p2 = stats.kendalltau(np.arange(len(fall)), -fall)
                if not np.isfinite(p1): p1 = 1.0
                if not np.isfinite(p2): p2 = 1.0
                # Fisher combination of one-sided trend evidence
                stat = -2*(np.log(max(p1,1e-12)) + np.log(max(p2,1e-12)))
                p_comb = float(stats.chi2.sf(stat, 4))
                score = -np.log10(max(p_comb, 1e-12))
                if best is None or p_comb < best["p_value"]:
                    fit_template = np.concatenate([np.linspace(np.min(yy), np.max(yy), peak_idx+1),
                                                   np.linspace(np.max(yy), np.min(yy), n-peak_idx)])
                    fitted_ordered = fit_template
                    fitted = np.empty_like(y, dtype=float)
                    fitted[np.array(order)] = fitted_ordered
                    best = {"period_h": float(p), "amplitude": float((np.nanmax(fitted)-np.nanmin(fitted))/2), "phase_deg": float((((phase/p)*360)+180)%360 - 180), "p_value": p_comb, "score": score, "notes": "Prespecified-period umbrella rise/fall scan", "fitted": fitted}
    return {"method":"RAIN", **best}

_FUNCS = {
    "Matrix pencil": matrix_pencil_method,
    "Cosinor": cosinor_method,
    "Harmonic regression": harmonic_regression_method,
    "Lomb–Scargle": lomb_scargle_method,
    "FFT": fft_method,
    "Wavelet": wavelet_method,
    "Autocorrelation": autocorrelation_method,
    "JTK-like": jtk_like_method,
    "ARSER-like": arser_like_method,
    "Bayesian harmonic": bayesian_rhythmic_method,
    "RAIN": rain_method,
}

def analyze_selected_methods(time_points, values, selected_methods=None):
    t, y = _as_arrays(time_points, values)
    selected = METHOD_ORDER if not selected_methods else [m for m in METHOD_ORDER if m in selected_methods]
    rows = []; fits = {}
    for name in selected:
        fn = _FUNCS[name]
        try:
            res = fn(t, y)
        except Exception as e:
            res = {"method": name, "period_h": np.nan, "amplitude": np.nan, "phase_deg": np.nan, "p_value": np.nan, "score": np.nan, "notes": f"Failed: {e}", "fitted": np.full_like(y, np.nan)}
        fits[name] = pd.DataFrame({"time": t, "observed": y, "fitted": res.get("fitted", np.full_like(y, np.nan))})
        rows.append({k:v for k,v in res.items() if k != "fitted"})
    return pd.DataFrame(rows), fits

def compare_methods_narrative(df):
    if df is None or df.empty: return "No method comparison results are available."
    ok = df.dropna(subset=["period_h"])
    if ok.empty: return "No method returned a finite dominant period."
    top = ok.sort_values(["score","amplitude"], ascending=[False, False]).head(4)
    bits = []
    for _, row in top.iterrows():
        pv = "" if pd.isna(row["p_value"]) else f", p={row['p_value']:.3g}"
        bits.append(f"{row['method']} estimated {row['period_h']:.3g} h with amplitude {row['amplitude']:.3g}{pv}")
    return "Top methods on this signal: " + "; ".join(bits) + "."
