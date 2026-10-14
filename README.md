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
saving to `<case>/results/`; `plot.py` redraws all figures in that folder.
For a small test, use `python3 portfolio/run.py --quick`.

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
| `--no-plot` | `run.py` | Save CSV results without drawing figures. |
| `--data PATH` | `plot.py` | Read CSV files from this folder and save redrawn figures there. |

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Explicit models, samplers, and CV² statistics |
| `experiments.py` | Parameter grids and independent repetitions |
| `plotting.py` | Shared figure styles and layouts |
| `<case>/run.py`, `<case>/plot.py` | Short entry points for each example |
| `diffusion/diffusion.py` | PDE solver, derivatives, and geometry |
| `<case>/results/` | CSV results and PNG/PDF figures |

Each case saves `n_comparison.csv` and `combined_N5000.csv`.
Ellipsoid also saves `d_lambda500.csv` and a coordinate illustration;
redraw it with `python3 upper_exterior_ellipsoid/schematic.py`.

## Reproducibility

Settings and random seeds are in `experiments.py`; algorithm details are in the paper.
Some archived combined results used an older random stream, so fresh runs may differ.
