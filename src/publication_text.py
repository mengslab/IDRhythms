
from __future__ import annotations
import pandas as pd

FIGURE_LEGEND_TEMPLATES = {
    "Overview panel": "Overview panel showing the observed signal, the highest-scoring fitted trace, dominant period estimates across methods, amplitude estimates across methods, and recommendation scores from the IDRhythms decision engine.",
    "Fitted gallery": "Fitted gallery showing the observed signal overlaid with fitted trajectories from all selected methods.",
    "Agreement heatmap": "Method-agreement heatmap summarizing dominant period, amplitude, phase, p-value surrogate, and method score across all selected methods.",
    "Period-amplitude map": "Period-amplitude summary map showing each method in period-amplitude space.",
    "Diagnostic panel": "Diagnostic panel showing FFT spectrum, autocorrelation profile, wavelet-power view, and phase-versus-score summary.",
}

def build_results_summary(result_df: pd.DataFrame, profile: dict, signal_name: str) -> str:
    if result_df is None or result_df.empty: return "No analysis results are available for summary."
    top = result_df.dropna(subset=["period_h"]).sort_values(["score","amplitude"], ascending=[False, False]).head(3)
    methods_text = "; ".join(f"{row.method} estimated a dominant period of {row.period_h:.2f} h with amplitude {row.amplitude:.3g} and phase {row.phase_deg:.2f}°" for row in top.itertuples())
    profile_bits = []
    if profile:
        profile_bits.append("evenly sampled" if profile.get("regular", False) else "unevenly sampled")
        if profile.get("short", False): profile_bits.append("short")
        if profile.get("noisy", False): profile_bits.append("noisy")
        if profile.get("damped", False): profile_bits.append("damped/nonstationary")
        if profile.get("asymmetric", False): profile_bits.append("asymmetric")
        if profile.get("multifrequency", False): profile_bits.append("multi-frequency")
    prefix = f"IDRhythms analysis of {signal_name} identified a signal profile that was " + ", ".join(profile_bits) + ". " if profile_bits else f"IDRhythms analysis of {signal_name} identified rhythmic structure. "
    return prefix + methods_text + "."

def build_methods_paragraph(selected_methods: list[str]) -> str:
    return ("Rhythmic analysis was performed using IDRhythms, a data-aware decision framework that integrates data validation, signal profiling, method recommendation, and multi-method inference. "
            "After preprocessing and signal-quality assessment, IDRhythms prioritized analytical methods based on sampling regularity, signal asymmetry, damping behavior, noise level, and multi-frequency content. "
            "The selected methods for this analysis were: " + ", ".join(selected_methods) + ".")

def build_publication_bundle(result_df: pd.DataFrame, profile: dict, signal_name: str, selected_methods: list[str], figure_names: list[str]) -> str:
    lines = [f"# IDRhythms publication bundle for {signal_name}", "", "## Results summary", build_results_summary(result_df, profile, signal_name), "", "## Methods text", build_methods_paragraph(selected_methods), "", "## Figure legends"]
    for name in figure_names:
        lines.append(f"- **{name}**: {FIGURE_LEGEND_TEMPLATES.get(name, f'{name}: IDRhythms publication figure.')}")
    return "\n".join(lines)
