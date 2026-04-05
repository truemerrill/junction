import jax
import jax.numpy as jnp

from .dynamics import omega_tau_matrix
from .problem import TransportProblem
from .types import (
    Array,
    ColoredNoiseFn,
    Matrix3,
    Matrix6,
    Matrix6Array,
    Scalar,
    TimeArray,
    WhiteNoiseFn,
)
from .unit import constants


def b_matrix() -> Array:
    return jnp.kron(
        jnp.array([[0], [1]], dtype=jnp.float64), jnp.eye(3, dtype=jnp.float64)
    )


def _quadrature_weights(time: TimeArray) -> TimeArray:
    """Construct trapezoidal-rule weights for a 1D time grid.

    Args:
        time (TimeArray): Monotone array of time points.

    Returns:
        TimeArray: Quadrature weights with the same length as ``time``.
    """
    if len(time) < 2:
        raise ValueError("Must provide at least two time points")

    dt = jnp.diff(time)
    w = jnp.zeros_like(time)
    w = w.at[0].set(0.5 * dt[0])
    w = w.at[-1].set(0.5 * dt[-1])
    if len(time) > 2:
        w = w.at[1:-1].set(0.5 * (dt[:-1] + dt[1:]))
    return w


def _symplectic_j() -> Matrix6:
    """Construct the canonical 6 x 6 symplectic matrix J."""
    I = jnp.eye(3, dtype=jnp.float64)
    Z = jnp.zeros((3, 3), dtype=jnp.float64)
    return jnp.block([[Z, I], [-I, Z]])


def _symplectic_inverse(U: Matrix6) -> Matrix6:
    """Compute the inverse of a symplectic 6 x 6 matrix.

    Uses
        U^{-1} = -J U^T J

    Args:
        U (Matrix6): Symplectic matrix.

    Returns:
        Matrix6: The inverse of ``U``.
    """
    J = _symplectic_j()
    return -J @ U.T @ J


def _noise_cov_matrix(
    time: TimeArray,
    noise_cov: ColoredNoiseFn,
) -> Array:
    """Construct the two-time noise covariance matrix.

    Args:
        time (TimeArray): Array of time points of shape (N,).
        noise_cov (NoiseCovarianceFn): Function mapping (s, s') to a
            3 x 3 covariance matrix.

    Returns:
        Array: Array of shape (N, N, 3, 3) where entry (i, j) is
            noise_cov(time[i], time[j]).
    """

    def row_fn(s: Scalar) -> Array:
        return jax.vmap(noise_cov, in_axes=(None, 0))(s, time)

    return jax.vmap(row_fn)(time)


def covariance_colored_noise(
    fundamental_matrix_soln: Matrix6Array, time: TimeArray, noise_cov: ColoredNoiseFn
) -> Matrix6Array:
    """Evaluate the colored-noise covariance at the final time.

    Computes the double integral

        Σ_x(t)
            = ∫₀ᵗ ∫₀ᵗ Φ(t, s) B G(s, s') Bᵀ Φ(t, s')ᵀ ds ds'

    using a trapezoidal quadrature over the provided time grid.

    This implementation assumes the input fundamental matrix is the
    principal solution

        U(t) = Φ(t, 0),

    and rewrites the integral as

        Σ_x(t)
            = U(t) [∫∫ C(s) G(s, s') C(s')ᵀ ds ds'] U(t)ᵀ,

    where

        C(s) = U(s)^{-1} B.

    Args:
        fundamental_matrix_soln (Matrix6Array): Array of shape (N, 6, 6)
            containing U(t_i) on the time grid.
        time (TimeArray): Array of shape (N,) containing time points.
        noise_cov (NoiseCovarianceFn): Function mapping (s, s') to a
            3 x 3 covariance matrix.

    Returns:
        Matrix6Array: The 6 x 6 covariance matrix at all sampled times.

    Raises:
        ValueError: If the time array length does not match the number of
        stored fundamental matrices.
    """
    if len(time) != fundamental_matrix_soln.shape[0]:
        raise ValueError("time and fundamental_matrix_soln must have same length")

    B = b_matrix()  # (6, 3)
    w = _quadrature_weights(time)  # (N,)

    U = fundamental_matrix_soln  # (N, 6, 6)

    # U^{-1}(t_i)
    U_inv = jax.vmap(_symplectic_inverse)(U)  # (N, 6, 6)

    # C_i = U^{-1}(t_i) B
    C = jnp.einsum("nij,jk->nik", U_inv, B)  # (N, 6, 3)

    # G_ij = noise_cov(t_i, t_j)
    G = _noise_cov_matrix(time, noise_cov)  # (N, N, 3, 3)

    # term_ij = C_i G_ij C_j^T
    term = jnp.einsum("iac,ijcd,jbd->ijab", C, G, C)  # (N, N, 6, 6)

    # Apply product quadrature weights.
    ww = jnp.outer(w, w)  # (N, N)
    weighted = jnp.einsum("ij,ijab->ijab", ww, term)  # (N, N, 6, 6)

    # Prefix double-integral:
    # inner_prefix[n, m] = sum_{i <= n, j <= m} weighted[i, j]
    inner_prefix = jnp.cumsum(jnp.cumsum(weighted, axis=0), axis=1)  # (N, N, 6, 6)

    # We want M_n = inner_prefix[n, n].
    n = jnp.arange(len(time))
    M = inner_prefix[n, n]  # (N, 6, 6)

    # Σ_n = U_n M_n U_n^T
    Sigma = jnp.einsum("nij,njk,nlk->nil", U, M, U)  # (N, 6, 6)

    return Sigma


def covariance_white_noise(
    fundamental_matrix_soln: Matrix6Array,
    time: TimeArray,
    noise_cov: WhiteNoiseFn,
) -> Matrix6Array:
    """Evaluate the white-noise covariance on the full sampled time grid.

    Computes

        Σ_x(t)
            = ∫₀ᵗ Φ(t, s) B Q(s) Bᵀ Φ(t, s)ᵀ ds

    at every sampled time point, where Q(s) is the instantaneous 3 x 3
    white-noise covariance and U(t) = Φ(t, 0).

    Using C(s) = U(s)^{-1} B, this becomes

        Σ_x(t)
            = U(t) [∫₀ᵗ C(s) Q(s) C(s)ᵀ ds] U(t)ᵀ,

    evaluated with trapezoidal quadrature on the provided grid.

    Args:
        fundamental_matrix_soln (Matrix6Array): Array of shape (N, 6, 6)
            containing U(t_i) on the time grid.
        time (TimeArray): Array of shape (N,) containing time points.
        noise_cov (WhiteNoiseFn): Function mapping s -> (3, 3) covariance Q(s).

    Returns:
        Matrix6Array: Array of shape (N, 6, 6) with Σ_x(t_i) at each time.

    Raises:
        ValueError: If the time array length does not match the number of
        stored fundamental matrices.
    """
    if len(time) != fundamental_matrix_soln.shape[0]:
        raise ValueError("time and fundamental_matrix_soln must have same length")

    B = b_matrix()  # (6, 3)
    w = _quadrature_weights(time)  # (N,)

    U = fundamental_matrix_soln  # (N, 6, 6)

    # U^{-1}(t_i)
    U_inv = jax.vmap(_symplectic_inverse)(U)  # (N, 6, 6)

    # C_i = U^{-1}(t_i) B
    C = jnp.einsum("nij,jk->nik", U_inv, B)  # (N, 6, 3)

    # Q_i = noise_cov(t_i)
    Q = jax.vmap(noise_cov)(time)  # (N, 3, 3)

    # integrand_i = C_i Q_i C_i^T
    integrand = jnp.einsum("nic,ncd,njd->nij", C, Q, C)  # (N, 6, 6)

    # M_n = ∫₀^{t_n} ... ds  (prefix integral via trapezoidal weights)
    weighted = w[:, None, None] * integrand  # (N, 6, 6)
    M = jnp.cumsum(weighted, axis=0)  # (N, 6, 6)

    # Σ_n = U_n M_n U_n^T
    Sigma = jnp.einsum("nij,njk,nlk->nil", U, M, U)  # (N, 6, 6)

    return Sigma


def white_noise(
    problem: TransportProblem, power_spectral_density: float = 1.0
) -> WhiteNoiseFn:
    """Construct a function computing the white-noise covariance

    Args:
        problem (TransportProblem): the transport problem
        power_spectral_density (float, optional): the electric noise power
            spectral density. Defaults to 1.0.

    Returns:
        WhiteNoiseFn: the white noise convolution kernel
    """
    q = problem.ion.charge
    m = problem.ion.mass
    Se = power_spectral_density
    omega_tau_inv = omega_tau_matrix(problem, power=-1.0)

    def white_noise_fn(s: Scalar) -> Matrix3:
        return (q**2 * Se) / (2 * m) * omega_tau_inv

    return white_noise_fn


def nbar(covariance_matrix: Matrix6Array, omega_tau: Matrix3) -> Array:
    """Calculate nbar from the covariance matrix for each mode

    Args:
        covariance_matrix (Matrix6Array): the covariance matrix
        omega_tau (Matrix3): the `\\Omega_\\tau` matrix

    Returns:
        Array: nbar for each mode
    """

    def e_matrix(index: int) -> Matrix3:
        return jnp.zeros((3, 3), dtype=jnp.float64).at[index, index].set(1.0)

    I2 = jnp.eye(2, dtype=jnp.float64)
    omegas = jnp.diag(omega_tau)
    thetas = [omegas[k] * jnp.kron(I2, e_matrix(k)) for k in range(3)]

    def nbar(theta: Matrix3, omega: Scalar) -> Array:
        p = theta @ covariance_matrix  # (N, 6, 6)
        return jnp.trace(p, axis1=-2, axis2=-1) / (2 * constants.hbar * omega)  # (N,)

    n = jnp.stack(
        [nbar(theta, omega) for theta, omega in zip(thetas, omegas)], axis=-1
    )  # (N, 3)
    return n
