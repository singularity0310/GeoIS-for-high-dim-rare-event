# GeoIS experiments

Four rare-event examples compare GeoIS, LSIS, and CE; diffusion also includes LAIS-dm.
Requires Python 3.10+.

## Run

From this folder:

```sh
python3 -m pip install -r requirements.txt
python3 portfolio/run.py --quick  # Small run; saves to a temporary folder.
python3 portfolio/run.py          # Full run; replaces this case's results.
python3 portfolio/plot.py         # Redraw saved figures without simulation.
```

Replace `portfolio` with `quadratic`, `upper_exterior_ellipsoid`, or `diffusion`.
Both scripts accept `--sweep n` or `--sweep combined`.
Run options: `--workers 4`, `--replicates 100`, `--output-dir PATH`, `--no-plot`.
Plot option: `--data PATH`.

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

## Reproduce

Full runs use 100 repetitions; `--quick` uses 3 with smaller parameter grids.
Settings and random seeds are in `experiments.py`; algorithm details are in the paper.
Some archived combined results used an older random stream, so fresh runs may differ.
