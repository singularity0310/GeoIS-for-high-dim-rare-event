"""Finite-element diffusion solves, adjoint gradients, and full tangent geometry."""

from functools import lru_cache
import time
import numpy as np
from scipy.linalg import eigh, cholesky_banded, cho_solve_banded
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigsh

HVP_STEP = 0.05
EIGENSOLVER_TOL = 1e-4


@lru_cache(maxsize=4)
def kl_system(elements=512, correlation_length=0.01, std_a=0.1):
    s = (np.arange(elements) + 0.5) / elements
    sigma2 = np.log1p(std_a**2)
    covariance = np.exp(-np.abs(s[:, None] - s[None, :]) / correlation_length)
    # Midpoint Nystrom quadrature gives weight 1/elements at each point.
    values, vectors = eigh(covariance / elements)
    values, vectors = values[::-1], vectors[:, ::-1]
    # Fix eigenvector signs for reproducibility.
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(elements)])
    vectors *= signs
    # beta[j, i] = sqrt(gamma_i) * e_i(s_j), including the field variance.
    beta = vectors * np.sqrt(elements * sigma2 * values)
    return s, -0.5 * sigma2, beta, values


class DiffusionPDE:
    def __init__(self, dimension=150, elements=512, std_a=0.1, correlation_length=0.01):
        if not 2 <= dimension <= elements:
            raise ValueError("Require 2 <= ambient KL dimension <= spatial elements")
        self.dimension, self.elements = dimension, elements
        self.s, self.mu, beta, self.kl_values = kl_system(elements, correlation_length, std_a)
        self.beta = beta[:, :dimension].copy()
        self.h = 1.0 / elements
        self.rhs = np.full(elements, self.h)
        self.rhs[-1] = self.h / 2.0
        self.output = np.zeros(elements)
        self.output[-1] = 1.0

    def _factor(self, u):
        """Assemble A = D.T diag(a/h) D and factor its band matrix."""
        a = np.exp(self.mu + self.beta @ u)
        stiffness = a / self.h
        diagonal = stiffness.copy()
        diagonal[:-1] += stiffness[1:]
        band = np.zeros((2, self.elements))
        # Reverse the nodes so elimination starts at the Neumann endpoint.
        band[0] = diagonal[::-1]
        band[1, :-1] = -stiffness[1:][::-1]
        factor = cholesky_banded(band, lower=True, check_finite=True)
        return factor, stiffness

    @staticmethod
    def _linear_solve(factor, rhs):
        return cho_solve_banded((factor, True), rhs[::-1], check_finite=False)[::-1]

    def solve(self, u):
        """Numerical forward solve A(u) v = f; return the endpoint v[-1]."""
        factor, _ = self._factor(u)
        state = self._linear_solve(factor, self.rhs)
        return float(state[-1])

    def value_gradient(self, u):
        """Solve Av=f and A.T p=c, then use F_i=-p.T A_i v."""
        factor, stiffness = self._factor(u)
        state = self._linear_solve(factor, self.rhs)
        adjoint = self._linear_solve(factor, self.output)
        # Differences between adjacent spatial nodes.
        dv = np.diff(state, prepend=0.0)
        dp = np.diff(adjoint, prepend=0.0)
        gradient = -self.beta.T @ (stiffness * dv * dp)
        return float(state[-1]), gradient

    def full_boundary_geometry(self, u, normal, ell):
        """Approximate the projected Hessian with HVPs; keep all tangent modes."""
        _, gradient = self.value_gradient(u)
        gradient_norm = np.linalg.norm(gradient)

        def matvec(v):
            # Project the direction onto the tangent space.
            tangent = v - normal * np.dot(normal, v)
            length = np.linalg.norm(tangent)
            if length < 1e-12:
                return np.zeros(self.dimension)
            direction = tangent / length
            # Forward-difference approximation to the Hessian-vector product.
            _, shifted_gradient = self.value_gradient(u + HVP_STEP * direction)
            hv = (shifted_gradient - gradient) / HVP_STEP
            projected = hv - normal * np.dot(normal, hv)
            return length * projected / gradient_norm

        operator = LinearOperator((self.dimension, self.dimension), matvec=matvec, dtype=float)
        indices = np.arange(self.dimension, dtype=float)
        initial = np.cos(0.37 * indices) + np.sin(0.11 * indices)
        initial -= normal * np.dot(normal, initial)
        initial /= np.linalg.norm(initial)
        values, vectors = eigsh(
            operator, k=self.dimension - 1, which="LM", tol=EIGENSOLVER_TOL, v0=initial
        )
        order = np.argsort(np.abs(values))[::-1]
        basis = vectors[:, order]
        # Remove normal leakage and orthonormalize without truncating directions.
        if np.linalg.norm(basis.T @ normal) > 1e-3:
            raise RuntimeError("Full HVP spectrum contains a normal mode")
        basis -= normal[:, None] * (normal @ basis)[None, :]
        basis, _ = np.linalg.qr(basis, mode="reduced")
        if (
            np.linalg.norm(basis.T @ normal) > 1e-8
            or np.linalg.norm(basis.T @ basis - np.eye(self.dimension - 1)) > 1e-8
        ):
            raise RuntimeError("Full HVP eigenbasis is not orthonormal and tangent")
        # B = -ell * Q.T Hessian(F) Q / norm(gradient(F)).
        return -ell * values[order], basis

    def log_response_from_field(self, log_a):
        """Solve the tridiagonal FE equations for batches of coefficient fields."""
        log_a = np.atleast_2d(log_a)
        values = []
        # Work with at most 2048 samples at a time to limit memory use.
        for start in range(0, len(log_a), 2048):
            stiffness = np.exp(log_a[start : start + 2048]).T.copy() / self.h
            diagonal = stiffness.copy()
            diagonal[:-1] += stiffness[1:]
            diagonal = diagonal[::-1].copy()
            off = -stiffness[1:][::-1]
            rhs = np.broadcast_to(self.rhs[::-1, None], diagonal.shape).copy()
            # Standard tridiagonal forward elimination and back substitution.
            for j in range(1, self.elements):
                multiplier = off[j - 1] / diagonal[j - 1]
                diagonal[j] -= multiplier * off[j - 1]
                rhs[j] -= multiplier * rhs[j - 1]
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
                raise FloatingPointError("FE stiffness factorization lost positive definiteness")
            rhs[-1] /= diagonal[-1]
            for j in range(self.elements - 2, -1, -1):
                rhs[j] = (rhs[j] - off[j] * rhs[j + 1]) / diagonal[j]
            state = rhs[::-1]
            response = state[-1]
            if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
                raise FloatingPointError("Invalid finite-element endpoint response")
            values.append(np.log(response))
        return np.concatenate(values)

    def log_response(self, u):
        return self.log_response_from_field(self.mu + np.atleast_2d(u) @ self.beta.T)

    def instanton_at_length(self, ell):
        """Solve u = ell * grad(F) / norm(grad(F)) by fixed-point iteration."""
        if not np.isfinite(ell) or ell <= 0:
            raise ValueError("ell must be finite and positive")
        _, g = self.value_gradient(np.zeros(self.dimension))
        u = ell * g / np.linalg.norm(g)
        for iteration in range(500):
            _, g = self.value_gradient(u)
            new = ell * g / np.linalg.norm(g)
            # Stop above the roundoff floor of the FE/adjoint solves.
            if np.linalg.norm(new - u) < 2e-11 * max(1.0, ell):
                u = new
                break
            u = new
        else:
            raise RuntimeError("Instanton sphere iteration did not converge")
        value, g = self.value_gradient(u)
        residual = np.linalg.norm(u / ell - g / np.linalg.norm(g))
        if residual > 1e-10:
            raise RuntimeError(f"Instanton stationarity residual: {residual}")
        return u, value, iteration + 1, residual

    def instanton_at_threshold(self, threshold):
        if threshold <= np.exp(self.log_response(np.zeros(self.dimension))[0]):
            raise ValueError("Threshold must exceed the deterministic response")

        def equation(ell):
            return self.instanton_at_length(ell)[1] - threshold

        hi = 1.0
        while equation(hi) < 0:
            hi *= 2.0
        ell = brentq(equation, 1e-9, hi, xtol=2e-13)
        return self.instanton_at_length(ell)


class DiffusionModel:
    """Map local samples (x,t) to KL coefficients and test the PDE event."""

    case = "diffusion"

    def __init__(
        self,
        d=20,
        lambda0=50.0,
        ell=1.0,
        threshold=None,
        elements=512,
        std_a=0.1,
        correlation_length=0.01,
    ):
        started = time.perf_counter()
        if d != int(d):
            raise ValueError("d must be an integer")
        if not np.isfinite(lambda0) or lambda0 <= 0:
            raise ValueError("lambda0 must be finite and positive")
        self.d, self.dimension, self.lambda0 = int(d), int(d) + 1, float(lambda0)
        self.pde = DiffusionPDE(self.dimension, elements, std_a, correlation_length)
        if threshold is None:
            self.u_star, self.threshold, self.iterations, self.kkt_residual = (
                self.pde.instanton_at_length(ell)
            )
        else:
            self.u_star, self.threshold, self.iterations, self.kkt_residual = (
                self.pde.instanton_at_threshold(threshold)
            )
        self.ell = float(np.linalg.norm(self.u_star))
        self.lam = self.lambda0 * self.ell**2
        self.normal = self.u_star / self.ell
        self.boundary_hessian, self.Q = self.pde.full_boundary_geometry(
            self.u_star, self.normal, self.ell
        )
        self.hessian_metric = 1.0 + self.boundary_hessian
        if np.any(self.hessian_metric <= 0):
            raise ValueError("H is not positive definite; no spectral clipping is allowed")
        self.field_tangent = self.ell * self.pde.beta @ self.Q
        self.field_normal = self.ell * self.pde.beta @ self.normal
        self.base_log_coefficient = self.pde.mu + self.pde.beta @ self.u_star
        self.log_threshold = np.log(self.threshold)
        self._last_samples = None
        self.response_count = 0
        self.setup_seconds = time.perf_counter() - started

    def original_coordinates(self, samples):
        # u = u_star + ell * (Qx + t * normal).
        return self.u_star + self.ell * (
            samples[:, :-1] @ self.Q.T + samples[:, -1, None] * self.normal
        )

    def quadratic_boundary(self, x):
        return 0.5 * np.sum(self.boundary_hessian * x * x, axis=1)

    def score(self, samples):
        # Cache the last batch; callers must not modify samples in place.
        if samples is not self._last_samples:
            log_a = (
                self.base_log_coefficient[None, :]
                + samples[:, :-1] @ self.field_tangent.T
                + samples[:, -1, None] * self.field_normal
            )
            self._last_score = self.log_threshold - self.pde.log_response_from_field(log_a)
            self._last_samples = samples
            self.response_count += len(samples)
        return self._last_score

    def event(self, samples):
        return self.score(samples) <= 0.0

    def target_logpdf(self, samples):
        norm2 = np.sum(samples[:, :-1] ** 2, axis=1) + (samples[:, -1] + 1.0) ** 2
        return 0.5 * self.dimension * np.log(self.lam / (2.0 * np.pi)) - 0.5 * self.lam * norm2

    def metadata(self):
        """Parameters and geometry saved with each result."""
        return {
            "d": self.d,
            "ambient_dimension": self.dimension,
            "lambda0": self.lambda0,
            "ell": self.ell,
            "lambda": self.lam,
            "threshold": self.threshold,
            "rank_B": int(np.linalg.matrix_rank(np.diag(self.boundary_hessian))),
            "retained_curvature_modes": self.d,
            "B_min": float(self.boundary_hessian.min()),
            "B_max": float(self.boundary_hessian.max()),
            "H_min": float(self.hessian_metric.min()),
            "H_max": float(self.hessian_metric.max()),
            "d2_over_lambda": self.d**2 / self.lam,
            "kkt_residual": float(self.kkt_residual),
            "spatial_elements": self.pde.elements,
            "response_solver": "P1 FEM / banded Cholesky and batched Thomas",
            "derivative_solver": (
                "discrete adjoint + forward-difference HVP / full eigsh spectrum"
            ),
            "hvp_step": HVP_STEP,
            "eigensolver_tolerance": EIGENSOLVER_TOL,
            "kl_variance_fraction": float(
                self.pde.kl_values[: self.dimension].sum() / self.pde.kl_values.sum()
            ),
            "field_normal_max": float(self.field_normal.max()),
            "setup_seconds": self.setup_seconds,
        }
