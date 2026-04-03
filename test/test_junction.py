import jax.numpy as jnp

from junction.dynamics import (
    b_matrix,
    fundamental_matrix,
    hessian_matrix,
    m_matrix,
    modes,
    omega_squared_matrix,
    omega_tau_matrix,
    s_matrix,
)
from junction.problem import problem
from junction.unit import constants as c

PROBLEM = problem()


def test_S_has_correct_shape():
    z = jnp.array(0.25)
    S_ = s_matrix(PROBLEM, z)
    assert S_.shape == (3, 3)


def test_S_is_orthogonal():
    z = jnp.array(0.25)
    S_ = s_matrix(PROBLEM, z)
    I = jnp.eye(3)
    assert jnp.allclose(S_.T @ S_, I)
    assert jnp.allclose(S_ @ S_.T, I)


def test_S_default_uses_z_stop():
    S_default = s_matrix(PROBLEM)
    S_stop = s_matrix(PROBLEM, PROBLEM.z_stop)
    assert jnp.allclose(S_default, S_stop)


def test_M_has_correct_shape():
    M_ = m_matrix(PROBLEM)
    assert M_.shape == (3, 3)


def test_M_is_diagonal():
    M_ = m_matrix(PROBLEM)
    assert jnp.allclose(M_, jnp.diag(jnp.diag(M_)))


def test_M_power_zero_is_identity():
    M0 = m_matrix(PROBLEM, power=0.0)
    assert jnp.allclose(M0, jnp.eye(3))


def test_M_power_one_matches_ion_mass_matrix():
    M1 = m_matrix(PROBLEM, power=1.0)
    expected = PROBLEM.ion.mass * jnp.eye(3)
    assert jnp.allclose(M1, expected)


def test_M_power_minus_one_inverts_M_power_one():
    M1 = m_matrix(PROBLEM, power=1.0)
    M_minus_1 = m_matrix(PROBLEM, power=-1.0)
    assert jnp.allclose(M1 @ M_minus_1, jnp.eye(3))
    assert jnp.allclose(M_minus_1 @ M1, jnp.eye(3))


def test_omega_tau_has_correct_shape():
    Omega = omega_tau_matrix(PROBLEM)
    assert Omega.shape == (3, 3)


def test_omega_tau_is_diagonal():
    Omega = omega_tau_matrix(PROBLEM)
    assert jnp.allclose(Omega, jnp.diag(jnp.diag(Omega)))


def test_omega_tau_power_zero_is_identity():
    Omega0 = omega_tau_matrix(PROBLEM, power=0.0)
    assert jnp.allclose(Omega0, jnp.eye(3))


def test_omega_tau_entries_match_final_angular_frequencies_matrix():
    z = PROBLEM.z_stop
    expected = jnp.diag(jnp.array([2 * jnp.pi * freq(z) for freq in PROBLEM.freqs]))
    actual = omega_tau_matrix(PROBLEM, power=1.0)
    assert jnp.allclose(actual, expected)


def test_K_has_correct_shape():
    z = jnp.array(0.25)
    K_ = hessian_matrix(PROBLEM, z)
    assert K_.shape == (3, 3)


def test_K_is_symmetric():
    z = jnp.array(0.25)
    K_ = hessian_matrix(PROBLEM, z)
    assert jnp.allclose(K_, K_.T)


def test_K_reduces_to_diagonal_in_mode_basis_matrix():
    z = PROBLEM.z_stop
    S_ = s_matrix(PROBLEM, z)
    K_ = hessian_matrix(PROBLEM, z)
    rotated = S_.T @ K_ @ S_
    assert jnp.allclose(rotated, jnp.diag(jnp.diag(rotated)))


def test_Omega2_has_correct_shape():
    z = jnp.array(0.25)
    Omega2_ = omega_squared_matrix(PROBLEM, z)
    assert Omega2_.shape == (3, 3)


def test_Omega2_is_symmetric():
    z = jnp.array(0.25)
    Omega2_ = omega_squared_matrix(PROBLEM, z)
    assert jnp.allclose(Omega2_, Omega2_.T)


def test_Omega2_is_diagonal_at_endpoint():
    z = PROBLEM.z_stop
    Omega2_ = omega_squared_matrix(PROBLEM, z)
    assert jnp.allclose(Omega2_, jnp.diag(jnp.diag(Omega2_)))


def test_modes_has_correct_shape():
    z = jnp.array(0.25)
    V = modes(PROBLEM, z)
    assert V.shape == (3, 3)


def test_modes_columns_have_expected_norms_matrix():
    z = jnp.array(0.25)
    scale = 1.7
    V = modes(PROBLEM, z, scale=scale)

    expected = jnp.array([scale * freq(z) for freq in PROBLEM.freqs])
    actual = jnp.linalg.norm(V, axis=0)

    assert jnp.allclose(actual, expected)


def test_modes_zero_scale_returns_zero_matrix():
    z = jnp.array(0.25)
    V = modes(PROBLEM, z, scale=0.0)
    assert jnp.allclose(V, jnp.zeros((3, 3)))


def test_modes_columns_are_orthogonal_when_normalized():
    z = jnp.array(0.25)
    V = modes(PROBLEM, z, scale=1.0)

    norms = jnp.linalg.norm(V, axis=0)
    U = V / norms[None, :]

    assert jnp.allclose(U.T @ U, jnp.eye(3))


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


def test_fundamental_matrix():
    time = jnp.linspace(0, 14, 1000, dtype=jnp.float64)
    U = fundamental_matrix(PROBLEM, time, dt=10 * c.nanosecond)
    assert U.shape == (1000, 6, 6)
    assert jnp.allclose(U[0, :, :], jnp.eye(6))
