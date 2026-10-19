# Rare-event experiments

This folder contains one retained parameter set for each of the four examples.
Each example has an N sweep and a three-column combined sweep. There is no
four-example overview or alternate-parameter sweep version.

## Fixed settings

| Setting | Quadratic, ellipsoid, portfolio | Diffusion |
| --- | --- | --- |
| Fixed ℓ outside the ℓ sweep | 0.5 | 0.5 |
| Fixed λ₀ outside the λ₀ sweep | 100 | 50 |
| Combined-sweep N | 5000 | 5000 |
| Independent repetitions | 100 | 100 |
| Random seed | 20260821 | 20260913 |
| N sweep dimensions | d=10 | d=10 and d=149 |
| N grid | 500:250:5000 | 500:500:5000 |
| λ₀ grid, with d=20 | 50:50:300 | 50, 100, 200, 300 |
| ℓ grid, with d=20 | 0.5:0.05:1 | 0.5:0.1:1 |
| d grid | 5:5:150 | 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150 |

The diffusion N sweep retains its original two dimensions; the combined
dimension sweep ends at d=150. All d values denote tangent dimensions.
Portfolio fixes ℓ, with z=exp(-ℓ/sqrt(d+1)); z therefore changes during the
dimension sweep.

## Results

Each `<case>/results/` contains:

- `n_comparison.csv` and `<case>_n_sweep.png/.pdf`.
- `combined_N5000.csv` and `<case>_combined.png/.pdf`.
- `manifest.json`, containing settings, data hashes, and provenance.

The combined figures use an independent y-axis for each CV² panel. All
methods, including LSIS and CE, are shown in the probability panels. In the
d-sweep probability panel, the lower y limit is −20 and the upper limit
adapts to the highest displayed estimate with a small margin. Curves extend
to and are clipped by the plot boundary; the underlying estimates are not
clamped or removed. Figure footnotes are omitted.
The CSV files retain every method. Ellipsoid has the same three columns as
the other examples. Horizontal axis padding is retained.

CV² definitions are unchanged: the explicit examples use pooled weight CV²
divided by N; diffusion plots independent-repetition CV² for every method,
including LAIS. At ℓ=1, the first three examples still have probability
estimates below 10⁻²⁰; the requested sweep endpoints are retained.

## Run or redraw

From this folder:

```sh
python3 -m pip install -r requirements.txt
python3 portfolio/plot.py          # Redraw both figures from saved CSVs.
python3 portfolio/run.py           # Recompute both sweeps and overwrite results.
```

Replace `portfolio` with `quadratic`, `upper_exterior_ellipsoid`, or `diffusion`.
The default run uses the settings above and four worker processes. There are
no alternate fixed-λ₀ or fixed-z profiles.

| Option | Applies to | Meaning |
| --- | --- | --- |
| `--sweep n` | Run and plot | N sweep only |
| `--sweep combined` | Run and plot | λ₀, ℓ, and d sweeps only |
| `--workers 4` | Run | Number of worker processes |
| `--replicates 100` | Run | Independent repetitions |
| `--output-dir PATH` | Run | Alternative results folder |
| `--no-plot` | Run | Save only data and metadata |
| `--quick` | Run | Small verification run with 3 repetitions, in a temporary folder |
| `--data PATH` | Plot | Folder containing the CSVs to redraw |

`main.py` contains the explicit models and sampling algorithms;
`diffusion/diffusion.py` contains the diffusion model;
`experiments.py` defines the retained grids and repetition runner;
`plotting.py` draws the N and combined figures.

The retained CSVs and figures were copied unchanged from the selected
results. Historical code hashes in each manifest document their origin;
deleted parameter versions are not required to run or redraw these results.

The ellipsoid coordinate illustration and `schematic.py` are separate from
the sweep experiments and remain available.
