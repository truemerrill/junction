import matplotlib.pyplot as plt
import jax.numpy as jnp

from junction.plots import plot_modes, plot_modes_path, hide_panes, plot_mode_frequencies, plot_mode_angles
from junction.problem import junction_problem


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