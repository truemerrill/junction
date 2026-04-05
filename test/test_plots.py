import jax.numpy as jnp
import matplotlib.pyplot as plt

from junction.covariance import covariance_white_noise, nbar, white_noise
from junction.dynamics import fundamental_matrix, omega_tau_matrix
from junction.plots import (
    hide_panes,
    modes,
    plot_mode_angles,
    plot_mode_frequencies,
    plot_modes,
    plot_modes_path,
)
from junction.problem import junction_problem, stationary_problem
from junction.unit import constants as c

PROBLEM = junction_problem()


def test_modes_has_correct_shape():
    z = jnp.array(0.25)
    V = modes(PROBLEM, z)
    assert V.shape == (3, 3)


def test_modes_columns_have_expected_norms_matrix():
    z = jnp.array(0.25)
    scale = 1.7
    V = modes(PROBLEM, z)

    expected = jnp.array([scale * freq(z) for freq in PROBLEM.freqs])
    actual = jnp.linalg.norm(V, axis=0)

    assert jnp.allclose(actual, expected)


def test_modes_columns_are_orthogonal():
    z = jnp.array(0.25)
    V = modes(PROBLEM, z)

    norms = jnp.linalg.norm(V, axis=0)
    U = V / norms[None, :]

    assert jnp.allclose(U.T @ U, jnp.eye(3))


def test_plot_mode_frequencies():
    problem = junction_problem()
    t = jnp.linspace(0, 14, 200)
    fig, ax = plt.subplots()
    plot_mode_frequencies(ax, problem, t)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Mode frequency (MHz)")
    plt.savefig("mode-frequencies.png")


def test_plot_mode_angles():
    problem = junction_problem()
    t = jnp.linspace(0, 14, 200)
    fig, ax = plt.subplots()
    plot_mode_angles(ax, problem, t)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Angle (deg)")
    plt.savefig("mode-angles.png")


def test_plot_modes_returns_three_artists_without_center():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    q = jnp.array([1.0, 2.0, 3.0])
    modes = jnp.eye(3)

    artists = plot_modes(ax, q, modes, center_marker="o")
    assert len(artists) == 4


def test_plot_stationary_heating_rate():
    time = jnp.linspace(0.0, 14.0, 1000, dtype=jnp.float64)

    # A stationary potential.
    problem = stationary_problem(freqs=jnp.array([1.0, 2.0, 3.0]))
    U = fundamental_matrix(problem, time, dt=1 * c.nanosecond)
    omega_tau = omega_tau_matrix(problem)
    G = white_noise(problem)

    Sigma = covariance_white_noise(U, time, G)
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
    plt.savefig("stationary-heating-rate.png")


def test_plot_modes_path():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = junction_problem()
    z = jnp.arange(0.44, 0.56, 0.01)

    plot_modes_path(ax, p, z, scale=2, center_marker="o", center_marker_size=3)
    hide_panes(ax, "x", "y", "z")
    ax.grid(False)
    ax.set_zticks([58, 64])
    ax.set_xlabel(r"x (µm)")
    ax.set_ylabel(r"y (µm)")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)

    plt.savefig("modes-path.png")
