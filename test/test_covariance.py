import jax.numpy as jnp
import matplotlib.pyplot as plt

from junction.covariance import (
    b_matrix,
    covariance_colored_noise,
    covariance_white_noise,
    nbar,
    white_noise,
)
from junction.dynamics import fundamental_matrix, omega_tau_matrix
from junction.problem import stationary_problem
from junction.types import Array, Matrix3
from junction.unit import constants as c

PROBLEM = stationary_problem()


def test_b_matrix_has_expected_shape():
    B = b_matrix()
    assert B.shape == (6, 3)


def test_b_matrix_has_expected_block_structure():
    B = b_matrix()

    assert jnp.allclose(B[:3, :], jnp.zeros((3, 3)))
    assert jnp.allclose(B[3:, :], jnp.eye(3))


def test_b_matrix_matches_kron_definition():
    B = b_matrix()
    expected = jnp.kron(
        jnp.array([[0], [1]], dtype=jnp.float64),
        jnp.eye(3, dtype=jnp.float64),
    )

    assert jnp.allclose(B, expected)


def test_b_matrix_maps_force_into_momentum_components_only():
    B = b_matrix()
    eta = jnp.array([1.2, -3.4, 5.6], dtype=jnp.float64)

    actual = B @ eta
    expected = jnp.array([0.0, 0.0, 0.0, 1.2, -3.4, 5.6], dtype=jnp.float64)

    assert jnp.allclose(actual, expected)


def test_b_matrix_transpose_times_b_is_identity():
    B = b_matrix()
    assert jnp.allclose(B.T @ B, jnp.eye(3))


def test_covariance_colored_noise_identity_fundamental_matrix():
    time = jnp.linspace(0.0, 14.0, 1000, dtype=jnp.float64)

    # Trivial dynamics: U(t) = I for all t, so Phi(t, s) = I.
    U = jnp.broadcast_to(jnp.eye(6, dtype=jnp.float64), (len(time), 6, 6))

    Q = jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))

    def noise_cov(s1, s2) -> Matrix3:  # type: ignore
        return Q

    Sigma = covariance_colored_noise(U, time, noise_cov)

    B = b_matrix()
    expected = (time[-1] ** 2) * (B @ Q @ B.T)

    assert Sigma.shape == (1000, 6, 6)
    assert jnp.allclose(Sigma[-1, :, :], expected, rtol=1e-3, atol=1e-3)


def test_covariance_stationary_heating_rate():
    time = jnp.linspace(0.0, 14.0, 1000, dtype=jnp.float64)

    # A stationary potential.
    problem = stationary_problem(freqs=jnp.array([1.0, 2.0, 3.0]) * c.MHz)
    U = fundamental_matrix(problem, time, dt=1 * c.nanosecond)
    omega_tau = omega_tau_matrix(problem)
    G = white_noise(problem, power_spectral_density=1.0)

    Sigma = covariance_white_noise(U, time, G)
    n = nbar(Sigma, omega_tau)

    # Measured slopes from linear least squares: n_j(t) ≈ a_j t + b_j
    A = jnp.stack([time, jnp.ones_like(time)], axis=1)  # shape (T, 2)

    def fit_slope(y: Array):
        coeffs, *_ = jnp.linalg.lstsq(A, y, rcond=None)
        slope, intercept = coeffs
        return slope, intercept

    fits = [fit_slope(n[:, j]) for j in range(3)]
    measured = jnp.array([f[0] for f in fits])

    # Theory
    q = problem.ion.charge
    m = problem.ion.mass
    hbar = c.hbar
    omega = jnp.diag(omega_tau)

    expected = q**2 / (4 * m * hbar * omega)

    # Ratio check (tighter, since ratios cancel prefactor error)
    assert jnp.allclose(
        measured / measured[0], expected / expected[0], rtol=5e-3, atol=1e-6
    )

    # Absolute slope check (looser due to timestep error)
    assert jnp.allclose(measured, expected, rtol=2e-2, atol=1e-6)
