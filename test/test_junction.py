import jax.numpy as jnp

from junction.problem import problem
from junction.dynamics import K, M, Omega2, Omega_tau, S, modes

PROBLEM = problem()


def test_S_has_correct_shape():
    z = jnp.array(0.25)
    S_ = S(PROBLEM, z)
    assert S_.shape == (3, 3)


def test_S_is_orthogonal():
    z = jnp.array(0.25)
    S_ = S(PROBLEM, z)
    I = jnp.eye(3)
    assert jnp.allclose(S_.T @ S_, I)
    assert jnp.allclose(S_ @ S_.T, I)


def test_S_default_uses_z_stop():
    S_default = S(PROBLEM)
    S_stop = S(PROBLEM, PROBLEM.z_stop)
    assert jnp.allclose(S_default, S_stop)


def test_M_has_correct_shape():
    M_ = M(PROBLEM)
    assert M_.shape == (3, 3)


def test_M_is_diagonal():
    M_ = M(PROBLEM)
    assert jnp.allclose(M_, jnp.diag(jnp.diag(M_)))


def test_M_power_zero_is_identity():
    M0 = M(PROBLEM, power=0.0)
    assert jnp.allclose(M0, jnp.eye(3))


def test_M_power_one_matches_ion_mass():
    M1 = M(PROBLEM, power=1.0)
    expected = PROBLEM.ion.mass * jnp.eye(3)
    assert jnp.allclose(M1, expected)


def test_M_power_minus_one_inverts_M_power_one():
    M1 = M(PROBLEM, power=1.0)
    M_minus_1 = M(PROBLEM, power=-1.0)
    assert jnp.allclose(M1 @ M_minus_1, jnp.eye(3))
    assert jnp.allclose(M_minus_1 @ M1, jnp.eye(3))


def test_Omega_tau_has_correct_shape():
    Omega = Omega_tau(PROBLEM)
    assert Omega.shape == (3, 3)


def test_Omega_tau_is_diagonal():
    Omega = Omega_tau(PROBLEM)
    assert jnp.allclose(Omega, jnp.diag(jnp.diag(Omega)))


def test_Omega_tau_power_zero_is_identity():
    Omega0 = Omega_tau(PROBLEM, power=0.0)
    assert jnp.allclose(Omega0, jnp.eye(3))


def test_Omega_tau_entries_match_final_angular_frequencies():
    z = PROBLEM.z_stop
    expected = jnp.diag(jnp.array([2 * jnp.pi * freq(z) for freq in PROBLEM.freqs]))
    actual = Omega_tau(PROBLEM, power=1.0)
    assert jnp.allclose(actual, expected)


def test_K_has_correct_shape():
    z = jnp.array(0.25)
    K_ = K(PROBLEM, z)
    assert K_.shape == (3, 3)


def test_K_is_symmetric():
    z = jnp.array(0.25)
    K_ = K(PROBLEM, z)
    assert jnp.allclose(K_, K_.T)


def test_K_reduces_to_diagonal_in_mode_basis():
    z = PROBLEM.z_stop
    S_ = S(PROBLEM, z)
    K_ = K(PROBLEM, z)
    rotated = S_.T @ K_ @ S_
    assert jnp.allclose(rotated, jnp.diag(jnp.diag(rotated)))


def test_Omega2_has_correct_shape():
    z = jnp.array(0.25)
    Omega2_ = Omega2(PROBLEM, z)
    assert Omega2_.shape == (3, 3)


def test_Omega2_is_symmetric():
    z = jnp.array(0.25)
    Omega2_ = Omega2(PROBLEM, z)
    assert jnp.allclose(Omega2_, Omega2_.T)


def test_Omega2_is_diagonal_at_endpoint():
    z = PROBLEM.z_stop
    Omega2_ = Omega2(PROBLEM, z)
    assert jnp.allclose(Omega2_, jnp.diag(jnp.diag(Omega2_)))


def test_modes_has_correct_shape():
    z = jnp.array(0.25)
    V = modes(PROBLEM, z)
    assert V.shape == (3, 3)


def test_modes_columns_have_expected_norms():
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
