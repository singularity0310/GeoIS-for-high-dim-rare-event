"""Draw the original u coordinates and the mapped quadratic boundary."""

from pathlib import Path
import os
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ellipse_schematic_cache"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ellipse_schematic_cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

M, ELL = 1.5, 1.0
BLUE, SHADE = "#17465f", "#d7eaf3"
RED, ORANGE = "#b53a44", "#d47b13"
MISSED = "#fde2bd"
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main():
    radius = np.sqrt(2) * M
    center = ELL - M
    limit = 1.35 * radius
    u1 = np.linspace(-limit, limit, 1801)
    upper = center + np.sqrt(np.maximum(M**2 - u1**2 / 2, 0))
    ymax = ELL + 0.90 * M
    ymin = center - 1.15 * M
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.fill_between(u1, upper, ymax, color=SHADE, zorder=0)

    angle = np.linspace(0, np.pi, 1001)
    ax.plot(radius * np.cos(angle), center + M * np.sin(angle), color=BLUE, lw=2.8, zorder=4)
    ax.plot(
        radius * np.cos(angle),
        center - M * np.sin(angle),
        color="#a6adb1",
        lw=1.4,
        ls=(0, (5, 4)),
        zorder=1,
    )
    # The event also has two horizontal boundary rays.
    ax.plot([-limit, -radius], [center, center], color=BLUE, lw=2.1)
    ax.plot([radius, limit], [center, center], color=BLUE, lw=2.1)
    ax.plot([-radius, radius], [center, center], color="#8b9499", lw=1.2, ls=(0, (2, 3)))

    # x=u1/ell and u2=ell*(1+t), with phi_hat(x)=-ell*x^2/(4m).
    mapped_phi_hat = ELL - u1**2 / (4 * M)
    # Event points below the parabola are missed by the upper proposal.
    ax.fill_between(
        u1,
        upper,
        mapped_phi_hat,
        where=mapped_phi_hat > upper,
        interpolate=True,
        color=MISSED,
        zorder=1,
    )
    ax.plot(u1, mapped_phi_hat, color=ORANGE, lw=2.5, ls=(0, (6, 4)), zorder=5)
    label_x = 0.97 * radius
    ax.annotate(
        r"$u_2=\ell+\ell\,\widehat\phi(u_1/\ell)$",
        (label_x, ELL - label_x**2 / (4 * M)),
        xytext=(1.23 * radius, ELL + 0.12 * M),
        ha="right",
        color=ORANGE,
        fontsize=16,
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1),
    )

    ax.axhline(0, color="#818a90", lw=0.9, zorder=2)
    ax.axvline(0, color="#818a90", lw=0.9, zorder=2)
    ax.plot(0, center, marker="+", color="#707a80", ms=9, mew=1.5, zorder=5)
    ax.plot(0, 0, marker="o", color="#20262b", ms=7, zorder=6)
    ax.plot(
        0,
        ELL,
        marker="*",
        color=RED,
        ms=16,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=7,
    )
    ax.annotate(
        "Gaussian mean\n" + r"$(0,0)$",
        (0, 0),
        xytext=(0.18 * radius, center - 0.13 * M),
        fontsize=13,
        va="top",
        arrowprops=dict(arrowstyle="-", color="#626d74", lw=0.9),
    )
    ax.annotate(
        r"$u^*=(0,\ell)$",
        (0, ELL),
        xytext=(0.13 * radius, ELL + 0.38 * M),
        color=RED,
        fontsize=14,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.9),
    )
    ax.text(-0.89 * radius, ELL + 0.60 * M, r"$D_0$", color=BLUE, fontsize=18)
    ax.annotate(
        r"$F(u)=z$",
        (-0.76 * radius, center + M * np.sqrt(1 - 0.76**2)),
        xytext=(-1.20 * radius, ELL + 0.09 * M),
        color=BLUE,
        fontsize=16,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=1),
    )
    ax.text(
        -1.24 * radius,
        center - 0.13 * M,
        r"$u_2=\ell-m$",
        fontsize=14,
        color="#626d74",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )

    ax.set(xlim=(-limit, limit), ylim=(ymin, ymax), aspect="equal")
    ax.set_xlabel(r"$u_1$", fontsize=18, labelpad=5)
    ax.set_ylabel(r"$u_2$", fontsize=18, rotation=0, labelpad=12)
    ax.set_xticks([-radius, 0, radius])
    ax.set_xticklabels([r"$-\sqrt{2}m$", r"$0$", r"$\sqrt{2}m$"], fontsize=14)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d1d6d9")
    ax.set_title("Original coordinates", fontsize=17, pad=16)
    ax.text(
        0.5,
        -0.18,
        r"$F(u)=\frac{1}{2}\left[\frac{1}{2}u_1^2+(u_2+m-\ell)^2\right]$"
        + r"$\qquad z=\frac{m^2}{2},\quad u_2\geq\ell-m$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
    )
    handles = [
        Patch(facecolor=SHADE, label="Upper-exterior event"),
        Patch(facecolor=MISSED, label="Missed by upper proposal"),
        Line2D([], [], color=BLUE, lw=2.8, label="Event boundary"),
        Line2D(
            [],
            [],
            color=ORANGE,
            lw=2.5,
            ls=(0, (6, 4)),
            label=r"Mapped quadratic $\widehat\phi$",
        ),
        Line2D([], [], color="#a6adb1", lw=1.4, ls="--", label="Lower ellipse"),
        Line2D([], [], marker="*", color=RED, ls="", ms=12, label="Instanton"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.5,
        handlelength=2.2,
        fontsize=12,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.82)
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    stem = output / "upper_exterior_ellipsoid_coordinates_u_xt"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Keep the paper's existing PDF path in sync with the case output.
    shutil.copy2(stem.with_suffix(".pdf"), output.parents[1] / (stem.name + ".pdf"))


if __name__ == "__main__":
    main()
