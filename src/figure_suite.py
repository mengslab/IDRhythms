
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal
import pywt

def apply_pub_layout(fig: go.Figure, x_title: str = "", y_title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#17313b"),
                      margin=dict(l=60, r=20, t=45, b=55),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.75)"))
    fig.update_xaxes(title=x_title, showline=True, linewidth=1.2, linecolor="black", mirror=True, ticks="outside", zeroline=False)
    fig.update_yaxes(title=y_title, showline=True, linewidth=1.2, linecolor="black", mirror=True, ticks="outside", zeroline=False)
    return fig

def figure_download_bundle(fig: go.Figure):
    payload = {"svg": None, "png": None, "html": None}
    try: payload["svg"] = fig.to_image(format="svg")
    except Exception: pass
    try: payload["png"] = fig.to_image(format="png", scale=2)
    except Exception: pass
    try: payload["html"] = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    except Exception: pass
    return payload

def _wavelet_power(time, values):
    t = np.asarray(time, dtype=float); y = np.asarray(values, dtype=float)
    if len(t) < 4: return None
    dt = np.median(np.diff(t))
    periods = np.linspace(max(2*dt, 2), max(min(48, (t[-1]-t[0])*0.95), 8), 100)
    scales = periods / (dt * pywt.central_frequency("morl"))
    coef, _ = pywt.cwt(y - np.mean(y), scales, "morl", sampling_period=dt)
    return periods, np.abs(coef) ** 2

def make_publication_figures(time, observed, result_df, fits_df_map, recommendation_df=None):
    figs = {}
    obs = np.asarray(observed, dtype=float); t = np.asarray(time, dtype=float)
    best_method = None
    if result_df is not None and not result_df.empty:
        best_method = result_df.sort_values(["score", "amplitude"], ascending=[False, False]).iloc[0]["method"]
    fit_df = fits_df_map.get(best_method) if best_method else None

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Observed vs best fit", "Dominant period by method", "Amplitude by method", "Recommendation scores"))
    fig.add_trace(go.Scatter(x=t, y=obs, mode="lines+markers", name="Observed", line=dict(width=2)), row=1, col=1)
    if fit_df is not None:
        fig.add_trace(go.Scatter(x=fit_df["time"], y=fit_df["fitted"], mode="lines", name=str(best_method), line=dict(width=2)), row=1, col=1)
    tmp = result_df.dropna(subset=["period_h"]) if result_df is not None else pd.DataFrame()
    if not tmp.empty: fig.add_trace(go.Bar(x=tmp["method"], y=tmp["period_h"], name="Period"), row=1, col=2)
    tmp = result_df.dropna(subset=["amplitude"]) if result_df is not None else pd.DataFrame()
    if not tmp.empty: fig.add_trace(go.Bar(x=tmp["method"], y=tmp["amplitude"], name="Amplitude"), row=2, col=1)
    if recommendation_df is not None and not recommendation_df.empty:
        fig.add_trace(go.Bar(x=recommendation_df["method"], y=recommendation_df["recommendation_score"], name="Recommendation"), row=2, col=2)
    fig.update_layout(height=780, paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
    figs["Overview panel"] = fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=obs, mode="lines+markers", name="Observed", line=dict(color="black", width=2)))
    for m, fdf in fits_df_map.items():
        fig.add_trace(go.Scatter(x=fdf["time"], y=fdf["fitted"], mode="lines", name=m, opacity=0.85))
    figs["Fitted gallery"] = apply_pub_layout(fig, "Time", "Signal", 480)

    heat_df = result_df.copy()
    heat_df["p_value_plot"] = heat_df["p_value"].fillna(1.0)
    heat_df["score_plot"] = heat_df["score"].fillna(0.0)
    heat_metrics = heat_df[["method","period_h","amplitude","phase_deg","p_value_plot","score_plot"]].set_index("method").T
    figs["Agreement heatmap"] = apply_pub_layout(px.imshow(heat_metrics.fillna(0), text_auto=".2g", aspect="auto", color_continuous_scale="Viridis", labels=dict(x="Method", y="Metric", color="Value")), "", "", 420)

    fig = px.scatter(result_df.dropna(subset=["period_h","amplitude"]), x="period_h", y="amplitude", color="method", text="method", hover_data=["phase_deg","score","notes"])
    fig.update_traces(textposition="top center")
    figs["Period-amplitude map"] = apply_pub_layout(fig, "Dominant period (h)", "Amplitude", 430)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("FFT spectrum", "Autocorrelation", "Wavelet power", "Phase vs score"))
    if len(t) >= 4:
        dt = np.median(np.diff(t)); tu = np.arange(t[0], t[-1] + dt * 0.5, dt); yu = np.interp(tu, t, obs)
        freqs = np.fft.rfftfreq(len(yu), d=dt); spec = np.abs(np.fft.rfft(yu - np.mean(yu)))
        periods = np.where(freqs > 0, 1 / np.maximum(freqs, 1e-9), np.nan); mask = np.isfinite(periods)
        fig.add_trace(go.Scatter(x=periods[mask], y=spec[mask], mode="lines", name="FFT spectrum"), row=1, col=1)
        acf = signal.correlate(yu - np.mean(yu), yu - np.mean(yu), mode="full"); acf = acf[len(acf)//2:]; acf = acf / max(acf[0], 1e-12)
        lags = np.arange(len(acf)) * dt
        fig.add_trace(go.Scatter(x=lags, y=acf, mode="lines", name="ACF"), row=1, col=2)
        w = _wavelet_power(tu, yu)
        if w is not None:
            wp, power = w
            fig.add_trace(go.Heatmap(x=tu, y=wp, z=power, colorscale="Viridis", showscale=False), row=2, col=1)
    tmp = result_df.dropna(subset=["phase_deg","score"])
    if not tmp.empty:
        fig.add_trace(go.Scatter(x=tmp["phase_deg"], y=tmp["score"], mode="markers+text", text=tmp["method"], textposition="top center"), row=2, col=2)
    fig.update_layout(height=820, paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
    figs["Diagnostic panel"] = fig
    return figs

def make_group_comparison_figure(curve_df: pd.DataFrame, title: str = "Control vs treated rhythmic fits"):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Group fits", "Parameter summary"))
    fig.add_trace(go.Scatter(x=curve_df["time"], y=curve_df["control"], mode="lines", name="Control"), row=1, col=1)
    fig.add_trace(go.Scatter(x=curve_df["time"], y=curve_df["treated"], mode="lines", name="Treated"), row=1, col=1)
    summary = curve_df.attrs.get("summary_table")
    if summary is not None and not summary.empty:
        fig.add_trace(go.Bar(x=summary["parameter"], y=summary["control"], name="Control"), row=1, col=2)
        fig.add_trace(go.Bar(x=summary["parameter"], y=summary["treated"], name="Treated"), row=1, col=2)
    fig.update_layout(title=title, height=430, paper_bgcolor="white", plot_bgcolor="white")
    return fig

def build_demo_signal_gallery(demo_df: pd.DataFrame):
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(demo_df.columns[1:5]))
    cols = list(demo_df.columns[1:5]); positions = [(1,1),(1,2),(2,1),(2,2)]
    for c,(r,cl) in zip(cols, positions):
        fig.add_trace(go.Scatter(x=demo_df["time"], y=demo_df[c], mode="lines+markers", name=c), row=r, col=cl)
    fig.update_layout(height=720, paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
    return fig
