# GeoIS experiments

Four rare-event examples compare GeoIS, LSIS, and CE; diffusion also includes LAIS-dm.
Requires Python 3.10+.

## Run

From this folder:

```sh
python3 -m pip install -r requirements.txt
python3 portfolio/run.py          # Simulate and overwrite this case's results.
python3 portfolio/plot.py         # Redraw figures from existing CSV files.
```

Replace `portfolio` with `quadratic`, `upper_exterior_ellipsoid`, or `diffusion`.
By default, `run.py` runs all sweeps with 4 processes and 100 independent repetitions,
saving to `<case>/results/`; `plot.py` redraws both sweep figures in that folder.
For a small test, use `python3 portfolio/run.py --quick`.

The fixed settings outside the corresponding sweeps are ℓ=0.5 and λ₀=100
(λ₀=50 for diffusion). Combined sweeps use N=5000, with λ₀ from 50 to 300,
ℓ from 0.5 to 1, and d from 5 to 150. Portfolio fixes ℓ in the d-sweep,
with z=exp(-ℓ/sqrt(d+1)). Exact grids are in `experiments.py`.

## Options

Omit options to use the defaults. Replace `PATH` with a folder path.

| Option | Script | Meaning |
| --- | --- | --- |
| `--quick` | `run.py` | Use smaller grids and sample sizes, with 3 repetitions; save to a temporary folder. |
| `--sweep n` | Both | Only process the sample-size N sweep. |
| `--sweep combined` | Both | Only process the precision, length, and dimension sweeps. |
| `--workers 4` | `run.py` | Use 4 parallel processes. |
| `--replicates 100` | `run.py` | Run 100 independent repetitions (R, not N); overrides the quick default. |
| `--output-dir PATH` | `run.py` | Save CSV results and figures to this folder. |
| `--no-plot` | `run.py` | Save results without drawing figures. |
| `--data PATH` | `plot.py` | Read CSV files from this folder and save redrawn figures there. |

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Explicit models, samplers, and CV² statistics |
| `experiments.py` | Parameter grids and independent repetitions |
| `plotting.py` | Shared figure styles and layouts |
| `<case>/run.py`, `<case>/plot.py` | Short entry points for each example |
| `diffusion/diffusion.py` | PDE solver, derivatives, and geometry |
| `<case>/results/` | CSV results, PNG/PDF figures, and a settings manifest |

Each case saves `n_comparison.csv`, `combined_N5000.csv`, and the corresponding
N-sweep and combined figures. Ellipsoid also includes a coordinate illustration;
redraw it with `python3 upper_exterior_ellipsoid/schematic.py`.

## Reproducibility

Settings and random seeds are in `experiments.py`; algorithm details are in the paper.
Each `manifest.json` records result settings and provenance. Some saved results used
earlier random streams, so fresh simulations may differ; redrawing the saved CSVs
reproduces the displayed curves.
