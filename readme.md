# LMAP: Local PCA Models with Global MDS Embeddings

LMAP (Local PCA Models with Global MDS Embeddings) is a compact Python library for nonlinear dimensionality reduction.  
It approximates a manifold using four transparent geometric components:

1. **Landmark sampling** – select representative points on the manifold.  
2. **Local PCA tangent modeling** – estimate local linear structure around each landmark.  
3. **Global MDS alignment** – align landmark charts via approximate geodesic distances.  
4. **Smooth out-of-sample extension** – embed all points by blending local linear predictions.

This yields an interpretable local–global coupling:  
local tangent models capture neighborhood geometry, while global MDS ensures coherent global structure.

---

## Installation

Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/evolution-strategies/lmap.git
cd lmap
pip install -r requirements.txt
```

Required packages include:
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`

---

## Quick Start: Swiss Roll Example

Run the example script:

```bash
cd examples
python swissroll.py
```

This:

- generates a 3D Swiss roll dataset,
- computes an LMAP embedding,
- prints trustworthiness (TW) and Sammon stress (SA),
- and visualizes both the embedding and the local tangent directions.

---

## Repository Structure

```
lmap/
│
├── lmap/
│   ├── lmap.py              # Core LMAP implementation
│   ├── metrics.py           # Trustworthiness (TW), Sammon stress (SA)
│   ├── datasets.py          # Standardized Swiss roll + utilities
│   ├── visualization.py     # 2D embedding + tangent plots
│
├── examples/
│   └── swissroll.py         # Minimal demonstration on Swiss roll
│
├── paper/
│   └── lmap.pdf             # Method description and evaluation
│
└── README.md
```

### Core modules:

- **`lmap/lmap.py`**  
  Main algorithm:
  - landmark selection  
  - local PCA tangent estimation  
  - landmark graph + shortest paths  
  - global MDS  
  - weighted blending for out-of-sample embedding

- **`lmap/metrics.py`**  
  Implementation of:
  - Trustworthiness (TW): local neighborhood preservation  
  - Sammon stress (SA): global distance distortion

- **`lmap/visualization.py`**  
  Plotting helpers for embeddings and local tangent directions.

- **`examples/`**  
  Contains runnable demo scripts.

---

## Paper

A detailed description of the method, algorithmic design, and experimental evaluation is provided in:

**`paper/lmap.pdf`**

The paper explains:
- the geometric motivation for LMAP,
- how local PCA models interact with global MDS,
- the influence of hyperparameters (`m`, `k_local`, `k_graph`),
- and comparisons against PCA, MDS, t-SNE, and UMAP.

---

## Citation

If you use LMAP in academic work, please cite:

> O. Kramer, *Local PCA Models with Global MDS Embeddings (LMAP)*, 2025.  
> (Full reference in `paper/lmap.pdf`.)
(paper under review at ESANN 2026)

---

## License

MIT License.