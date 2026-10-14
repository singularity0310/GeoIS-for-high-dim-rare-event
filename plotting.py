"""One publication figure per sweep, with only leading-term references."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rare_event_n5000_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rare_event_n5000_cache")
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = (
    "GeoIS",
    "GeoIS w=1",
    "GeoIS w=0.9",
    "LSIS",
    "CE round 1",
    "CE round 2",
    "CE round 3",
    "CE round 10",
    "LAIS-dm r=3",
    "LAIS-dm r=5",
)
STYLES = {
    "GeoIS": ("#1769aa", "o", "-"),
    "GeoIS w=1": ("#56B4E9", "D", "--"),
    "GeoIS w=0.9": ("#4292c6", "s", "--"),
    "LSIS": ("#888888", "s", "-"),
    "CE round 1": ("#74c476", "^", "--"),
    "CE round 2": ("#41ab5d", "v", "-."),
    "CE round 3": ("#238b45", "d", ":"),
    "CE round 10": ("#005a32", "P", "-"),
    "LAIS-dm r=3": ("#fdae6b", "*", "-"),
    "LAIS-dm r=5": ("#f16913", "X", "-"),
}
LABELS = {
    "GeoIS": r"GeoIS ($w=0.5$)",
    "GeoIS w=1": r"GeoIS ($w=1$)",
    "GeoIS w=0.9": r"GeoIS ($w=0.9$)",
    "LSIS": "LSIS (CE round 0)",
    "LAIS-dm r=3": r"LAIS-dm ($r=3$, $J=5$)",
    "LAIS-dm r=5": r"LAIS-dm ($r=5$, $J=5$)",
}
SWEEPS = ("lambda0", "ell", "d")
AXIS = {
    "lambda0": r"original precision $\lambda_0$",
    "ell": r"instanton length $\ell$",
    "d": r"tangent dimension $d$",
}
SWEEP_LABEL = {
    "lambda0": r"$\lambda_0$ sweep",
    "ell": r"$\ell$ sweep",
    "d": r"$d$ sweep",
}
plt.rcParams.update({"font.size": 14, "pdf.fonttype": 42, "ps.fonttype": 42})


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def fixed(rows, field):
    values = np.array([float(r[field]) for r in rows])
    assert np.allclose(values, values[0], rtol=1e-10, atol=1e-10), (field, values)
    return values[0]


def title(rows, case, sweep=None):
    fields = []
    if sweep != "d":
        fields.append(("d", "d"))
    if sweep != "lambda0":
        fields.append(("lambda0", r"\lambda_0"))
    if sweep != "ell":
        fields.append(("threshold", "z") if case == "portfolio" else ("ell", r"\ell"))
    if sweep is not None:
        fields.append(("N", "N"))
    params = r",\ ".join(f"{label}={fixed(rows, key):g}" for key, label in fields)
    return (SWEEP_LABEL[sweep] + "\n" if sweep else "") + "$" + params + "$"


def draw(ax, rows, ykey, xkey, markevery):
    handles = {}
    log_p = ykey == "log_mean_estimate"
    diffusion = rows[0]["case"] == "diffusion"
    for method in ORDER:
        selected = method_rows(rows, method, xkey)
        if not selected:
            continue
        x = np.array([float(r[xkey]) for r in selected])
        y = np.array([float(r[ykey]) for r in selected])
        # Stored log probabilities are natural logs; display base 10.
        if log_p:
            y = y / np.log(10.0)
        valid = np.isfinite(y)
        if ykey in ("cv2", "replicate_cv2"):
            valid &= y > 0
        # Leave gaps for undefined values or zero CV on a logarithmic axis.
        if not np.all(valid):
            print(f"{method}: omit {np.count_nonzero(~valid)} unplottable {ykey} points")
        y = np.where(valid, y, np.nan)
        color, marker, ls = STYLES[method]
        geo = method.startswith("GeoIS")
        width = 3.0 if geo else 1.35
        size = 4
        if geo:
            size = 5
        elif method.startswith("LAIS"):
            size = 6
        order = 5 if geo else 3
        mark_at = markevery
        face, edge_width = color, 0.7
        # Keep the dark GeoIS curve visible at overlaps.
        if method == "GeoIS" and log_p and rows[0]["sweep"] == "d":
            width, size, order = 1.35, 2.5, 6
        # Keep GeoIS visible beneath LAIS lines.
        if diffusion and log_p and geo:
            width, order = (3.5, 6) if method == "GeoIS" else (5.5, 5)
        # Alternate LAIS dashes and hollow markers at overlaps.
        if diffusion and log_p and method.startswith("LAIS"):
            offset = 0 if method == "LAIS-dm r=3" else 1
            width, size, order = 1.1, 7, 7
            ls = (4 * offset, (4, 4))
            mark_at = (offset, 2)
            face, edge_width = "none", 1.1
        # Hollow LSIS markers reveal overlapping curves.
        if method == "LSIS" and log_p:
            width, size, order = 0.8, 2.5, 9
            face, edge_width = "none", 0.5
            mark_at = 2
        # Use slightly heavier strokes in every N sweep.
        if xkey == "N":
            width += 0.45
        line = ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=width,
            markersize=size,
            markerfacecolor=face,
            markeredgewidth=edge_width,
            markevery=mark_at,
            zorder=order,
            label=LABELS.get(method, method),
        )[0]
        handles[method] = line
    return handles


def method_rows(rows, method, xkey):
    return sorted((r for r in rows if r["method"] == method), key=lambda r: float(r[xkey]))


def leading(ax, rows, xkey):
    """Draw the single 1/N reference, with no reference for w=1."""
    base = method_rows(rows, "GeoIS", xkey)
    x = np.array([float(r[xkey]) for r in base])
    n = np.array([int(r["N"]) for r in base])
    line = ax.plot(
        x,
        1 / n,
        color="#000000",
        linestyle=(0, (5, 3)),
        linewidth=2.25 if xkey == "N" else 1.8,
        zorder=8,
        label=r"$1/N$",
    )[0]
    return {"leading_GeoIS": line}


def decorate(ax, cv=False):
    if cv:
        ax.set_yscale("log")
    ax.grid(axis="y", which="major", color="#dddddd", alpha=0.6, linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_axisbelow(True)


def legend(fig, handles, ncol, fontsize):
    ordered = [v for k, v in handles.items() if k.startswith("leading_")]
    ordered += [handles[k] for k in ORDER if k in handles]
    # Keep both LAIS entries together in the last legend column.
    if ncol == 4 and "LAIS-dm r=3" in handles and "LAIS-dm r=5" in handles:
        keys = [
            "leading_GeoIS",
            "GeoIS",
            "GeoIS w=0.9",
            "LSIS",
            "CE round 1",
            "CE round 2",
            "CE round 3",
            "CE round 10",
            None,
            "LAIS-dm r=3",
            "LAIS-dm r=5",
            None,
        ]
        ordered = [
            handles[k] if k in handles else plt.Line2D([], [], linestyle="", label="") for k in keys
        ]
    return fig.legend(
        ordered,
        [h.get_label() for h in ordered],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        columnspacing=0.8,
        handletextpad=0.5,
        labelspacing=0.6,
    )


def output(fig, stem):
    """Save the PNG and PDF beside the experiment data."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.legends:
        bounds = item.get_window_extent(renderer)
        assert bounds.x0 >= -1 and bounds.x1 <= fig.bbox.x1 + 1
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return paths


def independent_cv_axis(ax, minimum_top):
    # Use this panel's data, with the largest tick at a power of ten.
    values = np.concatenate([line.get_ydata() for line in ax.lines])
    values = values[np.isfinite(values) & (values > 0)]
    top = max(minimum_top, 10.0 ** np.ceil(np.log10(values.max())))
    ax.set_ylim(ax.get_ylim()[0], top * 1.15)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=12))
    ax.tick_params(labelleft=True)


def n_sweep(rows, stem, cv2_key="cv2"):
    rows = [r for r in rows if int(r["N"]) <= 5000]
    case = rows[0]["case"]
    dims = sorted({int(r["d"]) for r in rows})
    diffusion = case == "diffusion"
    fig, axes = plt.subplots(
        1,
        len(dims),
        figsize=(14, 6.6) if diffusion else (9.5, 7.3),
        squeeze=False,
        sharey=False,
    )
    handles = {}
    for ax, d in zip(axes.flat, dims):
        selected = [r for r in rows if int(r["d"]) == d]
        handles.update(draw(ax, selected, cv2_key, "N", 1 if diffusion else 4))
        handles.update(leading(ax, selected, "N"))
        decorate(ax, True)
        if diffusion:
            independent_cv_axis(ax, minimum_top=1.0)
        ax.tick_params(labelsize=15)
        ax.set_title(title(selected, case), fontsize=18, pad=14)
        ax.set_xlabel(r"$N$ (samples per CE round / evaluation)", fontsize=17, labelpad=9)
    axes[0, 0].set_ylabel(r"$\mathrm{CV}^2$", fontsize=21, labelpad=8)
    legend(fig, handles, 4, 14)
    fig.subplots_adjust(
        left=0.08 if diffusion else 0.12,
        right=0.98,
        bottom=0.15,
        top=0.71 if diffusion else 0.79,
        wspace=0.16,
    )
    return output(fig, stem)


def combined(rows, stem, cv2_key="cv2", extra_d_rows=None):
    case = rows[0]["case"]
    columns = [(sweep, [r for r in rows if r["sweep"] == sweep]) for sweep in SWEEPS]
    if extra_d_rows is not None:
        columns.append(("d", [r for r in extra_d_rows if r["sweep"] == "d"]))
    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(14.5 + 4.5 * (ncols - 3), 9.2), sharex="col")
    handles = {}
    for col, (sweep, selected) in enumerate(columns):
        ax = axes[0, col]
        handles.update(draw(ax, selected, cv2_key, "x_value", 1))
        handles.update(leading(ax, selected, "x_value"))
        draw(axes[1, col], selected, "log_mean_estimate", "x_value", 1)
        ax.set_title(title(selected, case, sweep), fontsize=15, pad=17)
        decorate(ax, True)
        decorate(axes[1, col])
        axes[1, col].set_xlabel(AXIS[sweep], fontsize=17, labelpad=8)
    if case in ("portfolio", "diffusion"):
        for ax in axes[0]:
            independent_cv_axis(ax, minimum_top=0.1)
    else:
        limits = [ax.get_ylim() for ax in axes[0]]
        for ax in axes[0]:
            ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
        for ax in axes[0, 1:]:
            ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel(r"$\mathrm{CV}^2$", fontsize=19)
    axes[1, 0].set_ylabel(r"$\log_{10}\widehat p$", fontsize=19)
    for ax, letter in zip(axes.flat, "abcdefgh"):
        ax.text(
            0,
            1.01,
            f"({letter})",
            transform=ax.transAxes,
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
    legend(fig, handles, 4, 13)
    fig.subplots_adjust(
        left=0.085 if ncols == 3 else 0.065,
        right=0.985,
        bottom=0.105,
        top=0.78,
        hspace=0.25,
        wspace=0.23,
    )
    return output(fig, stem)


def cv2_column(rows):
    # Recycled DM samples are dependent; use independent-run variability.
    return "replicate_cv2" if any(r["method"].startswith("LAIS") for r in rows) else "cv2"


def plot_case(case, data_dir, sweep="all"):
    """Redraw only the selected paper figures from saved simulation data."""
    data = Path(data_dir)
    paths = []
    if sweep in ("all", "n"):
        rows = read(data / "n_comparison.csv")
        paths += n_sweep(rows, data / f"{case}_n_sweep", cv2_column(rows))
    if sweep in ("all", "combined"):
        rows = read(data / "combined_N5000.csv")
        extra = read(data / "d_lambda500.csv") if case == "upper_exterior_ellipsoid" else None
        paths += combined(rows, data / f"{case}_combined", cv2_column(rows), extra_d_rows=extra)
    return paths


def plot_case_cli(case, case_dir):
    import argparse

    parser = argparse.ArgumentParser(description="Redraw the saved N and combined figures.")
    parser.add_argument("--data", type=Path, default=Path(case_dir) / "results")
    parser.add_argument("--sweep", choices=("all", "n", "combined"), default="all")
    args = parser.parse_args()
    for path in plot_case(case, args.data, args.sweep):
        print(path)
