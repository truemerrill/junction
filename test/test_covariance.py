import jax.numpy as jnp
import matplotlib.pyplot as plt

from junction.covariance import b_matrix, covariance_colored_noise, covariance_white_noise, nbar
from junction.dynamics import fundamental_matrix, omega_tau_matrix
from junction.problem import stationary_problem
from junction.unit import constants as c
from junction.types import Matrix3



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



def test_covariance_identity_fundamental_matrix_2():
    time = jnp.linspace(0.0, 14.0, 1000, dtype=jnp.float64)

    # Trivial dynamics: U(t) = I for all t, so Phi(t, s) = I.
    U = fundamental_matrix(PROBLEM, time, dt=10 * c.nanosecond)
    Q = jnp.eye(3, dtype=jnp.float64) / 3
    omega_tau = omega_tau_matrix(PROBLEM)

    def noise_cov(s1) -> Matrix3:  # type: ignore
        return Q

    Sigma = covariance_white_noise(U, time, noise_cov)
    n = nbar(Sigma, omega_tau)

    fig, ax = plt.subplots()
    ax.plot(time, n[:, 0], label=r"$\omega_1$")
    ax.plot(time, n[:, 1], label=r"$\omega_2$")
    ax.plot(time, n[:, 2], label=r"$\omega_3$")
    ax.grid()
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel(r"$\bar{n} / S_E$")
    ax.legend()
    plt.show()

    B = b_matrix()
    expected = (time[-1] ** 2) * (B @ Q @ B.T)

    assert Sigma.shape == (1000, 6, 6)
    assert jnp.allclose(Sigma[-1, :, :], expected, rtol=1e-3, atol=1e-3)