
IDRhythms is a data-aware computational framework for robust rhythmicity analysis in biological time-series data. It integrates multiple complementary algorithms with a unified workflow for discovery, comparison, benchmarking, and validation of oscillatory signals.
This validation release (v2) establishes IDRhythms as a foundation platform for publication and grant applications, combining methodological rigor with reproducible outputs and interpretable visualizations.

Key Features

Multi-method rhythmicity detection
IDRhythms integrates faithful implementations of:
Matrix Pencil (damped oscillatory decomposition)
JTK-like nonparametric rhythmicity detection
ARSER-like harmonic regression
RAIN (robust detection of asymmetric rhythms)
These methods are applied within a unified framework to enable cross-method validation and consensus inference.

Data-aware method recommendation
The platform automatically profiles input data based on:
Sampling regularity
Noise level
Damping behavior
Asymmetry
Multi-frequency structure
and recommends optimal analysis strategies accordingly.

Benchmarking and validation engine

This release introduces a full validation layer:
Curated benchmark datasets (synthetic + realistic signals)
Expected outputs with tolerance ranges
Cross-method reference comparisons
Automated validation report generation
This enables objective assessment of method performance and reproducibility.

Publication-grade outputs
IDRhythms produces ready-to-use materials for manuscripts:
Multi-panel publication-quality figures
Figure legends (auto-generated)
Results summaries
Methods text
Exportable SVG / PNG / HTML figures

Control vs treated statistical framework
Effect size estimation (Hedges’ g)
Permutation / parametric tests
Phase-aware comparisons
FDR correction (Benjamini–Hochberg)

Design Philosophy
IDRhythms is built on three core principles:
Method complementarity — no single algorithm is sufficient
Data-aware inference — analysis must adapt to signal properties
Validation-first design — results must be benchmarked and reproducible

Use Cases
Circadian and ultradian rhythm analysis
Metabolic and transcriptomic time-series
Single-cell + bulk integration (future extensions)
Method benchmarking and algorithm development
Publication-ready figure generation

Roadmap
Bayesian and uncertainty-aware rhythm inference
Single-cell + bulk deconvolution integration (BayesRhythm)
Multi-condition trajectory-aware analysis
AI-assisted model selection and interpretation


# IDRhythms v2

IDRhythms v2 reintroduces Matrix pencil, JTK-like, ARSER-like, and RAIN into the main workflow together with the previously available methods.

## Included methods
- Matrix pencil
- Cosinor
- Harmonic regression
- Lomb–Scargle
- FFT
- Wavelet
- Autocorrelation
- JTK-like
- ARSER-like
- Bayesian harmonic
- RAIN

## Run
```bash
cd IDRhythms_v2
./run.sh
```
