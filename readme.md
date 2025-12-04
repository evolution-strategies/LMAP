# LMAP: Local PCA Models with Global MDS Embeddings

LMAP (Local PCA Models with Global MDS Embeddings) is a small reference
implementation of the method described in:

> **LMAP: Local PCA Models with Global MDS Embeddings**  
> Oliver Kramer, University of Oldenburg

LMAP is a geometry-based framework for nonlinear dimensionality reduction that:
- fits **local PCA tangent models** around landmark points,
- aligns these charts via **global MDS** on a landmark graph,
- and uses a **closed-form out-of-sample mapping** by blending local tangents.

This yields smooth, interpretable embeddings with explicit local–global structure coupling.

---

## Installation

```bash
pip install -e .