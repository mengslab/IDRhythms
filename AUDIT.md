
# IDRhythms v2 audit note

This build reintroduces Matrix pencil, JTK-like, ARSER-like, and RAIN into the main workflow.

Implementation notes:
- Matrix pencil: SVD-reduced matrix pencil on a Hankel embedding of an evenly resampled series, followed by pole extraction and dominant oscillatory component selection.
- JTK-like: period/phase grid scan with rank-based Kendall tau against reference rhythmic templates.
- ARSER-like: Yule-Walker autoregressive spectral estimation followed by harmonic regression at the dominant AR spectrum period.
- RAIN: prespecified-period umbrella rise/fall scan using rank-based monotonic trend tests combined across rise and fall segments.

These implementations are designed to be transparent, self-contained, and usable inside the local app.
