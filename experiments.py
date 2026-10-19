"""Paper settings and a common runner; all samplers live in main.py."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import logsumexp
from main import (
    Model,
    CASE_IDS,
    geo_is,
    simulate_ce_path,
    sample_lais_dm,
    seed_rng,
    aggregate_estimates,
    pooled_estimator_cv2,
)

FIXED_LAMBDA0 = {
    "quadratic": 100.0,
    "upper_exterior_ellipsoid": 100.0,
    "portfolio": 100.0,
    "diffusion": 50.0,
}
CE_ROUNDS = (1, 2, 3, 10)
LAIS_RANKS = (3, 5)
LAIS_J = 5


def specifications(case, quick=False):
    """Build the N sweep and the columns of the combined figure."""
    fixed_lambda0 = FIXED_LAMBDA0[case]
    n_dimensions = (10,)
    combined_n = 5000
    grids = {
        "n": list(range(500, 5001, 250)),
        "lambda0": list(range(50, 301, 50)),
        "ell": np.linspace(0.5, 1.0, 11).round(2).tolist(),
        "d": list(range(5, 151, 5)),
    }
    if case == "diffusion":
        n_dimensions = (10, 149)
        grids["n"] = list(range(500, 5001, 500))
        grids["lambda0"] = [50, 100, 200, 300]
        grids["ell"] = np.linspace(0.5, 1.0, 6).round(2).tolist()
        grids["d"] = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
    if quick:
        combined_n = 200
        grids = {"n": [100, 200], "lambda0": [50, 100], "ell": [0.5, 1.0], "d": [5, 20]}
        if case == "diffusion":
            grids["d"] = [5, 149]

    points = []
    for sweep, values in grids.items():
        dimensions = n_dimensions if sweep == "n" else (20,)
        for dimension in dimensions:
            for value in values:
                d = int(value) if sweep == "d" else dimension
                precision = fixed_lambda0
                ell = 0.5
                n = combined_n
                table = "combined_N5000.csv"

                if sweep == "n":
                    n = int(value)
                    table = "n_comparison.csv"
                elif sweep == "lambda0":
                    precision = float(value)
                elif sweep == "ell":
                    ell = float(value)

                points.append(
                    {
                        "case": case,
                        "sweep": sweep,
                        "x_value": value,
                        "d": d,
                        "lambda0": precision,
                        "ell": ell,
                        "N": n,
                        "table": table,
                    }
                )
    return points


def simulate(job):
    """Run one point with fresh CE and LAIS fits in every repetition."""
    spec, reps, seed = job
    case, n = spec["case"], spec["N"]
    diffusion = case == "diffusion"
    if diffusion:
        from diffusion.diffusion import DiffusionModel

        model = DiffusionModel(d=spec["d"], lambda0=spec["lambda0"], ell=spec["ell"])
        parts = (913, model.d, n, round(model.lambda0 * 1e6), round(model.ell * 1e9))
    else:
        model = Model(case, d=spec["d"], lambda0=spec["lambda0"], ell=spec["ell"])
        parts = (
            703,
            CASE_IDS[case],
            model.d,
            n,
            round(model.lambda0 * 1000),
            round(model.ell * 1e6),
        )
    extra = ("GeoIS w=1", 1.0, 2) if case == "quadratic" else ("GeoIS w=0.9", 0.9, 3)
    mixtures = [("GeoIS", 0.5, 1), extra]
    methods = [m for m, _, _ in mixtures] + ["LSIS"] + [f"CE round {r}" for r in CE_ROUNDS]
    if diffusion:
        methods += [f"LAIS-dm r={r}" for r in LAIS_RANKS]
    logs = {m: np.empty((reps, n)) for m in methods}
    log_scales = {}
    hits = {m: [] for m in methods}
    for i in range(reps):
        results = simulate_ce_path(model, n, seed_rng(seed, *parts, i, 0), CE_ROUNDS)
        for method, weight, stream in mixtures:
            results[method] = geo_is(model, n, seed_rng(seed, *parts, i, stream), weight)
        if diffusion:
            for rank in LAIS_RANKS:
                # Preserve the original random streams for both LAIS ranks.
                rank_seed = () if rank == 5 else (rank,)
                results[f"LAIS-dm r={rank}"] = sample_lais_dm(
                    model, n, seed_rng(seed, *parts, i, 2, *rank_seed), r=rank, j=LAIS_J
                )
        for method, result in results.items():
            # Pool W_i because the fixed SORM scale cancels from CV^2.
            logs[method][i] = result.log_weights
            if i == 0:
                log_scales[method] = result.log_scale
            elif result.log_scale != log_scales[method]:
                raise RuntimeError("The weight scale must be fixed across repetitions")
            hits[method].append(result.hit_rate)
    if diffusion:
        # Eleven CE/LSIS batches, two GeoIS batches, and two LAIS budgets.
        if model.response_count != reps * n * 15:
            raise RuntimeError("Unexpected number of PDE response evaluations")

    rows = []
    metadata = model.metadata()
    mixture_weights = {method: weight for method, weight, _ in mixtures}
    for method in methods:
        values = logs[method]
        # Restore p_SORM before averaging probabilities across repetitions.
        estimates = logsumexp(values, axis=1) - np.log(n) + log_scales[method]
        mean_log, rep_cv2 = aggregate_estimates(estimates)
        pooled_cv2 = pooled_estimator_cv2(values, n)
        ce_round = int(method.split()[-1]) if method.startswith("CE") else 0
        lais = method.startswith("LAIS")
        row = metadata.copy()
        row.update(
            case=case,
            fixed_ell=0.5,
            sweep=spec["sweep"],
            x_value=spec["x_value"],
            method=method,
            N=n,
            replicates=reps,
            seed=seed,
            log_mean_estimate=mean_log,
            cv2=rep_cv2 if lais else pooled_cv2,
            replicate_cv2=rep_cv2,
            mean_hit_rate=float(np.mean(hits[method])),
            ce_round=ce_round,
            ce_pilots_per_round=n if ce_round else 0,
            ce_total_pilots=ce_round * n,
            total_unique_samples_per_replicate=(ce_round + 1) * n,
            geo_mixture_weight=mixture_weights.get(method, ""),
            zero_estimate_replicates=int(np.count_nonzero(~np.isfinite(estimates))),
        )
        if lais:
            row.update(
                lais_rank=int(method.split("=")[-1]),
                lais_batches=LAIS_J,
                lais_n_ce=n // LAIS_J,
                lais_max_updates=LAIS_J - 1,
                lais_weighting="deterministic mixture",
            )
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case, case_dir):
    parser = argparse.ArgumentParser(description=f"Run the {case} paper experiments.")
    parser.add_argument("--quick", action="store_true", help="Small check in a temporary folder")
    parser.add_argument("--sweep", choices=("all", "n", "combined"), default="all")
    parser.add_argument("--replicates", type=int, help="Default: 100; quick: 3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    reps = 3 if args.quick else 100
    if args.replicates is not None:
        reps = args.replicates
    if reps < 2 or args.workers < 1:
        parser.error("Use at least two repetitions and one worker")
    output = Path(case_dir) / "results"
    if args.quick:
        output = Path(tempfile.gettempdir()) / f"{case}_quick_results"
    if args.output_dir is not None:
        output = args.output_dir
    specs = specifications(case, args.quick)
    if args.sweep == "n":
        specs = [s for s in specs if s["sweep"] == "n"]
    elif args.sweep == "combined":
        specs = [s for s in specs if s["sweep"] != "n"]
    seed = 20260913 if case == "diffusion" else 20260821
    jobs = [(spec, reps, seed) for spec in specs]
    tables = {}
    # Run each parameter point independently.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (spec, rows) in enumerate(zip(specs, pool.map(simulate, jobs)), 1):
            tables.setdefault(spec["table"], []).extend(rows)
            print(
                f"{index}/{len(specs)}: {spec['sweep']}={spec['x_value']}, d={spec['d']}",
                flush=True,
            )
    for name, rows in tables.items():
        write_csv(output / name, rows)
    write_manifest(case, output, tables)
    if not args.no_plot:
        from plotting import plot_case

        plot_case(case, output, args.sweep)
    print(f"Saved results in {output}")


def write_manifest(case, output, updated_tables):
    """Record new tables without discarding provenance for an unchanged sweep."""
    path = output / "manifest.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(
        case=case, fixed_ell=0.5, fixed_lambda0=FIXED_LAMBDA0[case],
        plot_options=dict(
            hide_ce_d_logp=False, independent_cv_axes=True, show_footnotes=False,
            d_logp_ylim=[-20, "auto"], d_logp_clipping="axis boundary",
        ),
        cv2_column="replicate_cv2" if case == "diffusion" else "cv2",
    )
    metadata.pop("replicates", None)
    metadata.pop("seed", None)
    tables = metadata.setdefault("tables", {})
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("main.py", "experiments.py", "diffusion/diffusion.py", "diffusion/__init__.py")}
    for name, rows in updated_tables.items():
        tables[name] = dict(
            sha256=hashlib.sha256((output / name).read_bytes()).hexdigest(),
            grids={sweep: sorted({float(row["x_value"]) for row in rows if row["sweep"] == sweep})
                   for sweep in sorted({row["sweep"] for row in rows})},
            dimensions=sorted({int(row["d"]) for row in rows}),
            replicates=rows[0]["replicates"], seed=rows[0]["seed"],
            sources=sources, note="Computed with the current code.",
        )
    path.write_text(json.dumps(metadata, indent=2) + "\n")
