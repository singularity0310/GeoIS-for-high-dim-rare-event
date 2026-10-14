"""Models and samplers in local coordinates: target N((0, -1), I / lambda)."""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.special import logsumexp

LOG_2PI = math.log(2.0 * math.pi)
CASES = ("portfolio", "quadratic", "upper_exterior_ellipsoid")
CASE_TITLES = {
    "portfolio": "Rescaled portfolio",
    "quadratic": "Full-rank exact quadratic",
    "upper_exterior_ellipsoid": "Shifted upper-exterior ellipsoid",
}
# Fixed IDs preserve the saved random streams.
CASE_IDS = {"portfolio": 0, "quadratic": 2, "upper_exterior_ellipsoid": 5}
UPPER_ELLIPSOID_A = 0.5
UPPER_ELLIPSOID_M = 1.5


@dataclass(frozen=True)
class Model:
    """Geometry of the three explicit examples; diffusion has its own model."""

    case: str
    d: int = 10
    lambda0: float = 100.0
    ell: float = 1.0
    quadratic_b0: float = 0.5

    def __post_init__(self):
        if self.case not in CASES:
            raise ValueError(f"case must be one of {CASES}")
        if not isinstance(self.d, (int, np.integer)) or self.d < 1:
            raise ValueError("d must be a positive integer")
        if not np.isfinite([self.lambda0, self.ell]).all() or min(self.lambda0, self.ell) <= 0:
            raise ValueError("lambda0 and ell must be finite and positive")

    @classmethod
    def portfolio_from_threshold(cls, d, lambda0, threshold):
        if not 0.0 < threshold < 1.0:
            raise ValueError("portfolio threshold must lie in (0, 1)")
        ell = math.sqrt(d + 1.0) * abs(math.log(threshold))
        return cls("portfolio", d=d, lambda0=lambda0, ell=ell)

    @property
    def title(self):
        return CASE_TITLES[self.case]

    @property
    def dimension(self):
        return self.d + 1

    @property
    def lam(self):
        return self.lambda0 * self.ell**2

    @property
    def portfolio_a(self):
        return self.ell / math.sqrt(self.d + 1.0)

    @property
    def portfolio_threshold(self):
        return math.exp(-self.portfolio_a)

    @property
    def upper_ellipsoid_rescaled_m(self):
        return UPPER_ELLIPSOID_M / self.ell

    @property
    def boundary_hessian(self):
        """Diagonal entries of B = Hessian(phi) at the origin."""
        if self.case == "portfolio":
            b = self.portfolio_a
        elif self.case == "quadratic":
            b = self.ell * self.quadratic_b0
        else:
            b = -UPPER_ELLIPSOID_A / self.upper_ellipsoid_rescaled_m
        return np.full(self.d, b)

    @property
    def hessian_metric(self):
        h = 1.0 + self.boundary_hessian
        if np.any(h <= 0.0):
            raise ValueError("H = I + B must be positive definite")
        return h

    def quadratic_boundary(self, x):
        """The proposal uses phi_hat(x) = x.T B x / 2."""
        return 0.5 * np.sum(self.boundary_hessian * np.asarray(x) ** 2, axis=1)

    def true_boundary(self, x):
        x = np.asarray(x)
        if self.case == "quadratic":
            return self.quadratic_boundary(x)
        if self.case == "portfolio":
            # Substitute the tangent coordinates into the portfolio sum.
            y = self.ell * (x @ tangent_basis(self.d + 1).T)
            return (logsumexp(y, axis=1) - math.log(self.d + 1.0)) / self.portfolio_a

        radius = 0.5 * np.sum(x * x, axis=1)
        m = self.upper_ellipsoid_rescaled_m
        # Outside the ellipse's projection, the boundary is the plane t=-m.
        return -m + np.sqrt(np.maximum(0.0, m**2 - radius))

    def event(self, samples):
        """Test the true boundary, keeping the original rounding at equality."""
        x = samples[:, :-1]
        t = samples[:, -1]
        if self.case in ("portfolio", "quadratic"):
            return t >= self.true_boundary(x)
        radius = 0.5 * np.sum(x * x, axis=1)
        m = self.upper_ellipsoid_rescaled_m
        return (t >= -m) & ((m + t) ** 2 + radius >= m**2)

    def score(self, samples):
        """CE uses the event score <= 0."""
        x = samples[:, :-1]
        t = samples[:, -1]
        if self.case in ("portfolio", "quadratic"):
            return self.true_boundary(x) - t
        radius = 0.5 * np.sum(x * x, axis=1)
        m = self.upper_ellipsoid_rescaled_m
        return np.maximum(-m - t, m**2 - (m + t) ** 2 - radius)

    def target_logpdf(self, samples):
        centered = np.asarray(samples).copy()
        centered[:, -1] += 1.0
        return 0.5 * self.dimension * (math.log(self.lam) - LOG_2PI) - 0.5 * self.lam * np.sum(
            centered * centered, axis=1
        )

    def metadata(self):
        """Problem parameters saved beside each estimate."""
        b = self.boundary_hessian
        h = self.hessian_metric
        row = {
            "case": self.case,
            "model_title": self.title,
            "d": self.d,
            "lambda0": self.lambda0,
            "ell": self.ell,
            "lambda": self.lam,
            "threshold": self.portfolio_threshold if self.case == "portfolio" else "",
            "rank_B": int(np.count_nonzero(b)),
            "B_min": float(np.min(b)),
            "B_max": float(np.max(b)),
            "H_min": float(np.min(h)),
            "H_max": float(np.max(h)),
            "theory_d2_over_lambda0": self.d**2 / self.lambda0,
            "theory_d3_over_lambda": self.d**3 / self.lam,
        }
        if self.case == "upper_exterior_ellipsoid":
            row["physical_m"] = UPPER_ELLIPSOID_M
            row["rescaled_m"] = self.upper_ellipsoid_rescaled_m
        return row


@lru_cache(maxsize=None)
def tangent_basis(dimension):
    """Helmert basis of the orthogonal complement of the all-ones vector."""
    basis = np.zeros((dimension, dimension - 1))
    for j in range(1, dimension):
        basis[:j, j - 1] = 1.0 / math.sqrt(j * (j + 1.0))
        basis[j, j - 1] = -j / math.sqrt(j * (j + 1.0))
    basis.setflags(write=False)
    return basis


@dataclass(frozen=True)
class GaussianProposal:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class SampleResult:
    """Log weights (-inf means zero); GeoIS adds log(p_SORM), other methods add 0."""

    log_weights: np.ndarray
    hit_rate: float
    log_scale: float = 0.0

    @property
    def log_contributions(self):
        """Full probability contributions, including the fixed scale."""
        return self.log_weights + self.log_scale

    @property
    def log_estimate(self):
        return float(logsumexp(self.log_weights) - math.log(len(self.log_weights)) + self.log_scale)


def _gaussian_logpdf(samples, mean, covariance):
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, (samples - mean).T).T
    return (
        -0.5 * samples.shape[1] * LOG_2PI
        - np.log(np.diag(chol)).sum()
        - 0.5 * np.sum(whitened * whitened, axis=1)
    )


def _draw_gaussian(proposal, count, rng):
    chol = np.linalg.cholesky(proposal.covariance)
    return rng.normal(size=(count, len(proposal.mean))) @ chol.T + proposal.mean


def _regularize(covariance, floor):
    """Keep covariance eigenvalues above the given floor."""
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _fit_gaussian(samples, log_weights):
    """Fit a weighted mean and covariance for CE and LAIS-dm."""
    weights = np.exp(log_weights - logsumexp(log_weights))
    mean = weights @ samples
    centered = samples - mean
    return mean, centered.T @ (weights[:, None] * centered)


def lsis_proposal(model):
    return GaussianProposal(np.zeros(model.dimension), np.eye(model.dimension) / model.lam)


def log_sorm_probability(model):
    """log(p_SORM), with p_SORM=exp(-lambda/2)/sqrt(2*pi*lambda*det(H))."""
    return float(
        -0.5 * model.lam
        - 0.5 * (LOG_2PI + math.log(model.lam))
        - 0.5 * np.sum(np.log(model.hessian_metric))
    )


def geo_is(model, count, rng, mixture_weight=0.5):
    """Algorithm 1: sample X, J, Y and return W_i with the fixed SORM scale."""
    if not 0.0 < mixture_weight <= 1.0:
        raise ValueError("mixture_weight must lie in (0,1]")
    if mixture_weight == 1.0 and model.case != "quadratic":
        raise ValueError("w=1 is only supported for the exact quadratic event")

    # Sample tangent coordinates, then the normal displacement.
    lam = model.lam
    h = model.hessian_metric
    x = rng.normal(size=(count, model.d)) / np.sqrt(lam * h)
    if mixture_weight == 1.0:
        upper = np.ones(count, dtype=bool)
    else:
        upper = rng.random(count) < mixture_weight
    gap = rng.exponential(scale=1.0 / lam, size=count)
    t = model.quadratic_boundary(x) + np.where(upper, gap, -gap)
    samples = np.column_stack((x, t))
    hit = model.event(samples)

    # w_i = J_i*w + (1-J_i)*(1-w), and gap is Y_i in Algorithm 1.
    branch_probability = np.where(upper, mixture_weight, 1.0 - mixture_weight)
    log_weight = (
        -0.5 * lam * t**2 + 2.0 * lam * np.where(upper, 0.0, gap) - np.log(branch_probability)
    )
    log_weight[~hit] = -math.inf
    return SampleResult(log_weight, float(np.mean(hit)), log_sorm_probability(model))


def evaluate_gaussian_batch(model, proposal, samples):
    hit = model.event(samples)
    logs = model.target_logpdf(samples) - _gaussian_logpdf(
        samples, proposal.mean, proposal.covariance
    )
    logs[~hit] = -np.inf
    return SampleResult(logs, float(hit.mean()))


def update_ce_proposal(model, proposal, samples):
    """Update the Gaussian mean and covariance with weighted elite samples."""
    score = model.score(samples)
    level = max(0.0, float(np.quantile(score, 0.10)))
    elite = score <= level
    minimum = min(len(samples) - 1, model.dimension + 2)
    if elite.sum() < minimum:
        elite = score <= np.partition(score, minimum - 1)[minimum - 1]
    selected = samples[elite]
    log_lr = (
        model.target_logpdf(samples) - _gaussian_logpdf(samples, proposal.mean, proposal.covariance)
    )[elite]
    mean, covariance = _fit_gaussian(selected, log_lr)
    return GaussianProposal(mean, _regularize(covariance, 0.05 / model.lam))


def simulate_ce_path(model, n, rng, saved_rounds=(1, 2, 3, 10)):
    """Refit CE each repetition; reuse each batch as the next round pilots."""
    if n < 2:
        raise ValueError("CE needs at least two samples per batch")
    proposal = lsis_proposal(model)
    samples = _draw_gaussian(proposal, n, rng)
    results = {"LSIS": evaluate_gaussian_batch(model, proposal, samples)}
    for r in range(1, max(saved_rounds) + 1):
        proposal = update_ce_proposal(model, proposal, samples)
        # Evaluate with q_r, then reuse this batch to fit q_(r+1).
        samples = _draw_gaussian(proposal, n, rng)
        if r in saved_rounds:
            results[f"CE round {r}"] = evaluate_gaussian_batch(model, proposal, samples)
    return results


def sample_lais_dm(model, n, rng, r=5, j=5):
    """LAIS-dm: use j equal batches and refit from all past weighted hits."""
    if not 1 <= r <= model.dimension or j < 1 or n < j or n % j:
        raise ValueError("Require 1 <= r <= dimension and N divisible by J, with N >= J >= 1")

    # Whitened coordinates are xi = sqrt(lambda) * (x, t+1).
    tangent = np.argsort(-np.abs(model.boundary_hessian), kind="stable")[: r - 1]
    # Include the normal and the r-1 largest tangent curvatures.
    active = np.concatenate(([model.d], tangent))
    mean = np.zeros(r)
    mean[0] = np.sqrt(model.lam)  # Start from LSIS.
    covariance = np.eye(r)
    size = n // j
    previous = np.empty((0, r))
    hits = np.empty(0, dtype=bool)
    log_mixture_sum = np.empty(0)
    proposals = []

    for step in range(j):
        covariance = _regularize(covariance, 1e-6)
        chol = np.linalg.cholesky(covariance)
        reduced = (chol @ rng.normal(size=(r, size))).T + mean
        xi = rng.normal(size=(size, model.dimension))
        xi[:, active] = reduced
        samples = xi / np.sqrt(model.lam)
        samples[:, -1] -= 1.0
        hit = model.event(samples)

        # Add the new proposal to the mixture density at every old sample.
        if len(previous):
            log_mixture_sum = np.logaddexp(
                log_mixture_sum,
                np.log(size) + _gaussian_logpdf(previous, mean, covariance),
            )
        proposals.append((size, mean.copy(), covariance.copy()))
        # Evaluate the count-weighted mixture at each new sample.
        terms = [np.log(count) + _gaussian_logpdf(reduced, mu, cov) for count, mu, cov in proposals]
        log_mixture_sum = np.concatenate((log_mixture_sum, logsumexp(terms, axis=0)))
        previous = np.vstack([previous, reduced])
        hits = np.concatenate((hits, hit))

        # q_mix = sum_j N_j q_j / N_total; recompute every sample's weight.
        log_target = -0.5 * (r * LOG_2PI + np.sum(previous**2, axis=1))
        log_weights = log_target + np.log(len(previous)) - log_mixture_sum
        log_weights[~hits] = -math.inf
        # Skip fitting when fewer than five event samples are available.
        if step < j - 1 and np.count_nonzero(hits) >= 5:
            mean, covariance = _fit_gaussian(previous, log_weights)

    return SampleResult(log_weights, float(np.mean(hits)))


def _scaled_weights(log_values):
    """Apply a common scale to avoid underflow; preserve zero weights."""
    values = np.asarray(log_values).ravel()
    if values.size < 2 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Need at least two log values, each finite or -inf")
    shift = float(np.max(values))
    if shift == -math.inf:
        return np.zeros(values.size), shift
    return np.exp(values - shift), shift


def aggregate_estimates(log_estimates):
    """Return log(mean estimate) and sample variance / mean^2 across repetitions."""
    scaled, shift = _scaled_weights(log_estimates)
    mean = float(np.mean(scaled))
    if mean == 0:
        return -math.inf, math.inf
    return shift + math.log(mean), float(np.var(scaled, ddof=1) / mean**2)


def pooled_estimator_cv2(log_weights, count):
    """Return S_W^2 / (N * mean(W)^2) from all RN weights."""
    if count < 1:
        raise ValueError("N must be positive")
    scaled, _ = _scaled_weights(log_weights)
    if not np.any(scaled):
        return math.inf
    return float(np.var(scaled, ddof=1) / (count * np.mean(scaled) ** 2))


def seed_rng(seed, *parts):
    """Create one reproducible random stream from fixed integer labels."""
    return np.random.default_rng(np.random.SeedSequence([seed, *parts]))


def self_check():
    rng = np.random.default_rng(20260821)
    for case in CASES:
        model = (
            Model.portfolio_from_threshold(4, 100.0, 0.8)
            if case == "portfolio"
            else Model(case, d=4, lambda0=100.0, ell=1.0)
        )
        if not np.all(model.hessian_metric > 0.0):
            raise AssertionError(f"non-positive H for {case}")
        result = geo_is(model, 1_000, rng)
        if not np.isfinite(result.log_estimate):
            raise AssertionError(f"GeoIS produced no finite estimate for {case}")
        print(f"{case:20s} log(p-hat)={result.log_estimate: .6f} hit={result.hit_rate:.3f}")


if __name__ == "__main__":
    self_check()
