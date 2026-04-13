from typing import Iterable, Literal

import jax.numpy as jnp
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .dynamics import s_matrix
from .problem import TransportProblem
from .types import Matrix3, Scalar, Vector
from .unit import constants as c


def plot_mode_frequencies(
    ax: Axes,
    problem: TransportProblem,
    t: Iterable[float],
    *,
    colors: tuple[str, str, str] = ("C0", "C1", "C2"),
) -> list[Line2D]:
    lines: list[Line2D] = []

    t = jnp.asarray(t)
    z = jnp.asarray([problem.waveform_index(tk) for tk in t])
    freqs = [[f(jnp.asarray(zk)) / c.MHz for zk in z] for f in problem.freqs]

    for freq, color in zip(freqs, colors):
        new_lines = ax.plot(t / c.microsecond, freq, color=color)
        lines.extend(new_lines)

    ax.grid()

    return lines


def plot_mode_angles(
    ax: Axes, problem: TransportProblem, t: Iterable[float]
) -> list[Line2D]:
    lines: list[Line2D] = []

    t = jnp.asarray(t)
    z = jnp.asarray([problem.waveform_index(tk) for tk in t])

    theta = jnp.asarray([problem.theta(zk) for zk in z])
    phi = jnp.asarray([problem.phi(zk) for zk in z])

    lines.extend(ax.plot(t / c.microsecond, theta, label=r"$\theta$"))
    lines.extend(ax.plot(t / c.microsecond, phi, label=r"$\phi$"))
    ax.grid()
    ax.legend()

    return lines


def modes(problem: TransportProblem, z: Scalar) -> Matrix3:
    """Construct mode vectors in the lab frame for visualization.

    Note:
        This function returns a (3, 3) matrix whose columns are the three
        principal-axis mode vectors at waveform index ``z``.

    Args:
        problem (TransportProblem): the transport problem.
        z (Scalar): the waveform index.

    Returns:
        Matrix: a (3, 3) array whose columns are the three mode vectors
        in the lab frame.
    """
    v = jnp.array([freq(z) for freq in problem.freqs])
    S_ = s_matrix(problem, z)
    return S_ @ jnp.diag(v)


def plot_modes(
    ax: Axes3D,
    q: Vector,
    modes: Matrix3,
    *,
    scale: float = 1.0,
    colors: tuple[str, str, str] = ("C0", "C1", "C2"),
    linewidths: float | tuple[float, float, float] = 2.0,
    linestyles: str | tuple[str, str, str] = "-",
    center_marker: str | None = None,
    center_marker_size: float = 6.0,
    center_color: str = "k",
    symmetric: bool = True,
    labels: tuple[str, str, str] | None = None,
    alpha: float = 1.0,
    zorder: float = 3,
) -> list[Artist]:
    """Plot a 3D mode-frame glyph centered at q.

    Note:
        The columns of ``modes`` are interpreted as the three mode vectors
        in the lab frame. By default, each mode is drawn symmetrically about
        the center point ``q``, so that the glyph looks like a 3D "jack".

    Args:
        ax (Axes): a 3D matplotlib axis.
        q (Vector): the center point, shape ``(3,)``.
        modes (Matrix3): mode vectors, shape ``(3, 3)``, with one vector per
            column.
        scale (float, optional): scale factor to scale the three arms.
        colors (tuple[str, str, str], optional): colors for the three arms.
            Defaults to ``("C0", "C1", "C2")``.
        linewidths (float | tuple[float, float, float], optional): line widths
            for the three arms. A scalar applies to all arms. Defaults to 2.0.
        linestyles (str | tuple[str, str, str], optional): line styles for the
            three arms. A scalar applies to all arms. Defaults to "-".
        center_marker (str | None, optional): marker style for the center point.
            If ``None``, no center marker is drawn. Defaults to None.
        center_marker_size (float, optional): size of the center marker.
            Defaults to 6.0.
        center_color (str, optional): color of the center marker. Defaults to
            "k".
        symmetric (bool, optional): if True, draw each arm from ``q - v`` to
            ``q + v``. If False, draw from ``q`` to ``q + v``. Defaults to True.
        labels (tuple[str, str, str] | None, optional): labels for the three
            arms. Useful for legends. If None, no labels are assigned.
            Defaults to None.
        alpha (float, optional): artist alpha. Defaults to 1.0.
        zorder (float, optional): z-order for the created artists. Defaults
            to 3.

    Returns:
        list[Artist]: the artists created by this function.
    """
    q = jnp.asarray(q, dtype=float).reshape(3)
    M = scale * jnp.asarray(modes, dtype=float).reshape(3, 3)

    def _triple(
        value: float | str | tuple[float, float, float] | tuple[str, str, str],
    ) -> tuple:  # type: ignore
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError("Expected a tuple of length 3.")
            return value
        return (value, value, value)

    linewidths_ = _triple(linewidths)
    linestyles_ = _triple(linestyles)

    if len(colors) != 3:
        raise ValueError("Expected exactly three colors.")
    if labels is not None and len(labels) != 3:
        raise ValueError("Expected exactly three labels.")

    artists: list[Artist] = []

    for i in range(3):
        v = M[:, i]

        if symmetric:
            p0 = q - v
            p1 = q + v
        else:
            p0 = q
            p1 = q + v

        label = None if labels is None else labels[i]

        artist = ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=colors[i],
            linewidth=linewidths_[i],
            linestyle=linestyles_[i],
            alpha=alpha,
            zorder=zorder,
            label=label,
        )[0]
        artists.append(artist)

    if center_marker is not None:
        center_artist = ax.plot(
            [q[0]],
            [q[1]],
            [q[2]],
            marker=center_marker,
            markersize=center_marker_size,
            color=center_color,
            alpha=alpha,
            zorder=zorder + 1,
            linestyle="None",
        )[0]
        artists.append(center_artist)

    return artists


def plot_modes_path(
    ax: Axes3D,
    problem: TransportProblem,
    z: Iterable[float],
    *,
    scale: float = 1.0,
    colors: tuple[str, str, str] = ("C0", "C1", "C2"),
    linewidths: float | tuple[float, float, float] = 2.0,
    linestyles: str | tuple[str, str, str] = "-",
    center_marker: str | None = None,
    center_marker_size: float = 6.0,
    center_color: str = "k",
    symmetric: bool = True,
    labels: tuple[str, str, str] | None = None,
    alpha: float = 1.0,
    zorder: float = 3,
) -> list[Artist]:
    """Plot a transport path of 3D mode-frame glyphs.

    Args:
        ax (Axes): a 3D matplotlib axis.
        problem (TransportProblem): the transport problem.
        z (Iterable[float]): the discrete points to render.
        scale (float, optional): scale factor to scale the three arms.
        colors (tuple[str, str, str], optional): colors for the three arms.
            Defaults to ``("C0", "C1", "C2")``.
        linewidths (float | tuple[float, float, float], optional): line widths
            for the three arms. A scalar applies to all arms. Defaults to 2.0.
        linestyles (str | tuple[str, str, str], optional): line styles for the
            three arms. A scalar applies to all arms. Defaults to "-".
        center_marker (str | None, optional): marker style for the center point.
            If ``None``, no center marker is drawn. Defaults to None.
        center_marker_size (float, optional): size of the center marker.
            Defaults to 6.0.
        center_color (str, optional): color of the center marker. Defaults to
            "k".
        symmetric (bool, optional): if True, draw each arm from ``q - v`` to
            ``q + v``. If False, draw from ``q`` to ``q + v``. Defaults to True.
        labels (tuple[str, str, str] | None, optional): labels for the three
            arms. Useful for legends. If None, no labels are assigned.
            Defaults to None.
        alpha (float, optional): artist alpha. Defaults to 1.0.
        zorder (float, optional): z-order for the created artists. Defaults
            to 3.

    Returns:
        list[Artist]: the artists created by this function.
    """

    artists: list[Artist] = []
    for zk in z:
        zk_ = jnp.asarray(zk)
        M = modes(problem, zk_)
        q = problem.qbar(zk_)
        new_artists = plot_modes(
            ax,
            q=q,
            modes=M,
            scale=scale,
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            center_marker=center_marker,
            center_marker_size=center_marker_size,
            center_color=center_color,
            symmetric=symmetric,
            labels=labels,
            alpha=alpha,
            zorder=zorder,
        )
        artists.extend(new_artists)

    ax.set_aspect("equal")
    return artists


def hide_panes(ax: Axes3D, *panes: Literal["x", "y", "z"]):
    """Hide the panes (back walls) in a 3D plot

    Args:
        ax (Axes3D): the axes
    """

    def hide_pane(pane: Polygon | None):
        if pane:
            pane.set_visible(False)

    def get_pane(axis: Axis) -> Polygon | None:
        if hasattr(axis, "pane"):
            pane = getattr(axis, "pane")
            if isinstance(pane, Polygon):
                return pane
        return None

    for label in panes:
        if label == "x":
            hide_pane(get_pane(ax.xaxis))
        if label == "y":
            hide_pane(get_pane(ax.yaxis))
        if label == "z":
            hide_pane(get_pane(ax.zaxis))
