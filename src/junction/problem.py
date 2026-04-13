from dataclasses import dataclass

import jax.numpy as jnp
from interpax import Interpolator1D

from . import data
from .species import CALCIUM_40, IonSpecies
from .types import ModeFreqs, ParameterFn, Scalar, Vector, VectorFn
from .unit import constants


@dataclass(frozen=True)
class TransportProblem:
    """Parameterization for a single-ion transport problem.

    Attributes:
        ion (IonSpecies): the ion species.
        qbar (VectorFn): function giving the equilibrium coordinate as a
            function of the waveform index.
        freqs (ModeFreqs): tuple of functions for the mode frequencies of the
            three principle axes.
        theta (ParameterFn): function describing the rotation angle about the
            y axis (in degrees) between the axial mode and the x axis.
        phi (ParameterFn): function describing the rotation angle about the
            x axis (in degrees) between the axial mode and the y axis.
        z_start (Scalar): the starting waveform index.
        z_stop (Scalar): the stopping waveform index.
    """

    ion: IonSpecies
    waveform_index: ParameterFn
    qbar: VectorFn
    freqs: ModeFreqs
    theta: ParameterFn
    phi: ParameterFn
    z_start: Scalar
    z_stop: Scalar


# -----------------------------------------------------------------------------
# Private functions, used to clean the data from W. C. Burton, et al. and to
# construct a TransportProblem from the interpolated data.


def _constant(x0: float | Scalar) -> ParameterFn:
    def constant_fn(s: Scalar) -> Scalar:
        return jnp.array(x0)

    return constant_fn


def _hold(fn: Interpolator1D) -> ParameterFn:
    x_min = fn.x.min()
    x_max = fn.x.max()

    def fn_hold(x: Scalar) -> Scalar:
        x_safe = jnp.clip(x, x_min, x_max)
        return fn(x_safe)

    return fn_hold


def _mirror(fn: ParameterFn, x0: float, x1: float) -> ParameterFn:
    """Mirror a parameter function about the midpoint of the unit interval.

    Constructs a new function defined on s ∈ [0, 1] that traverses the domain
    [max, min] on the first half (s ∈ [0, 0.5]) and then reverses direction,
    traversing [min, max] on the second half (s ∈ (0.5, 1]). This produces a
    time-symmetric (palindromic) parameterization, useful for enforcing
    time-reversal symmetry in waveform construction.

    Args:
        fn (ParameterFn): the function to mirror
        x0 (float): the starting value of the parameter at s = 0.
        x1 (float): the midpoint value of the parameter at s = 0.5

    Returns:
        ParameterFn: a new function g(s) = fn(x(s)) where x(s) follows a
            mirrored (forward-then-reverse) path between max and min.
    """
    fn_hold = _hold(fn) if isinstance(fn, Interpolator1D) else fn

    def reflect(s: Scalar) -> Scalar:
        left = jnp.asarray(x0 + (x1 - x0) * s / 0.5, dtype=jnp.float64)
        right = jnp.asarray(x1 + (x0 - x1) * (s - 0.5) / 0.5, dtype=jnp.float64)
        return jnp.where(s < 0.5, left, right)

    def mirrored_fn(s: Scalar) -> Scalar:
        x = reflect(s)
        return fn_hold(x)

    return mirrored_fn


def _hold_after_midpoint(fn: ParameterFn, x0: float, x1: float) -> ParameterFn:
    """Follow fn on the first half, then hold its midpoint value."""
    fn_hold = _hold(fn) if isinstance(fn, Interpolator1D) else fn
    half = jnp.asarray(0.5, dtype=jnp.float64)
    x1_arr = jnp.asarray(x1, dtype=jnp.float64)

    def forward_coord(s: Scalar) -> Scalar:
        return jnp.asarray(x0 + (x1 - x0) * s / 0.5, dtype=jnp.float64)

    def theta_fn(s: Scalar) -> Scalar:
        left = jnp.asarray(fn_hold(forward_coord(s)), dtype=jnp.float64)
        right = jnp.asarray(fn_hold(x1_arr), dtype=jnp.float64)
        return jnp.where(s <= half, left, right)

    return theta_fn


def _reverse_replay_after_midpoint(
    fn: ParameterFn,
    x0: float,
    x1: float,
    sign: float = -1.0,
) -> ParameterFn:
    """Keep zero on the first half, then replay fn in reverse after midpoint.

    For s > 1/2 this returns

        sign * (fn(x1) - fn(x_rev(s)))

    so the function is continuous at s = 1/2 and starts from zero there.
    """
    fn_hold = _hold(fn) if isinstance(fn, Interpolator1D) else fn
    half = jnp.asarray(0.5, dtype=jnp.float64)
    x1_arr = jnp.asarray(x1, dtype=jnp.float64)

    def reverse_coord(s: Scalar) -> Scalar:
        return jnp.asarray(x1 + (x0 - x1) * (s - 0.5) / 0.5, dtype=jnp.float64)

    def phi_fn(s: Scalar) -> Scalar:
        left = jnp.asarray(0.0, dtype=jnp.float64)
        right = jnp.asarray(
            sign * (fn_hold(x1_arr) - fn_hold(reverse_coord(s))),
            dtype=jnp.float64,
        )
        return jnp.where(s <= half, left, right)

    return phi_fn


def _ramp(x0: float, x1: float, x2: float) -> ParameterFn:
    """Construct a piecewise-linear ramp over the unit interval.

    Defines a function r(s) for s ∈ [0, 1] that linearly interpolates between
    three values (x0 → x1 → x2) with a breakpoint at s = 0.5:

        s ∈ [0, 0.5]   →   x0 → x1
        s ∈ (0.5, 1]   →   x1 → x2

    This is useful for building simple, differentiable parameter schedules
    with a single interior knot, e.g. specifying boundary values and a
    midpoint constraint.

    Args:
        x0 (float): Value at s = 0 (start of interval).
        x1 (float): Value at s = 0.5 (midpoint).
        x2 (float): Value at s = 1 (end of interval).

    Returns:
        ParameterFn: A scalar-valued function r(s) that performs the
            piecewise-linear interpolation x0 → x1 → x2 over s ∈ [0, 1].
    """

    def r(s: Scalar) -> Scalar:
        left = jnp.asarray(x0 + (x1 - x0) * (s / 0.5), dtype=jnp.float64)
        right = jnp.asarray(x1 + (x2 - x1) * ((s - 0.5) / 0.5), dtype=jnp.float64)
        return jnp.where(s < 0.5, left, right)

    return r


def stationary_problem(
    ion: IonSpecies = CALCIUM_40,
    freqs: Vector = jnp.array(
        [
            1.1 * constants.MHz,
            1.8 * constants.MHz,
            2.0 * constants.MHz,
        ]
    ),
) -> TransportProblem:
    """Construct a problem representing a stationary potential.

    Args:
        ion (IonSpecies, optional): the ion. Defaults to CALCIUM_40.
        freqs (Vector, optional): the mode frequencies. Defaults to
            [1.1, 1.8, 2.0] MHz.

    Returns:
        TransportProblem: the transport problem
    """

    def waveform_index(t: Scalar) -> Scalar:
        return jnp.array(0.0)

    def qbar(z: Scalar) -> Scalar:
        return jnp.array([0, 0, 0], dtype=jnp.float64)

    return TransportProblem(
        ion=ion,
        waveform_index=waveform_index,
        qbar=qbar,
        freqs=(_constant(freqs[0]), _constant(freqs[1]), _constant(freqs[2])),
        theta=_constant(jnp.array(0.0)),
        phi=_constant(jnp.array(0.0)),
        z_start=jnp.array(0.0),
        z_stop=jnp.array(1.0),
    )


def junction_problem(
    speed: float = 50 * constants.meter / constants.second, ion: IonSpecies = CALCIUM_40
) -> TransportProblem:
    """Construct a junction transport problem from Burton et al. data.

    This function builds a ``TransportProblem`` using interpolated trajectory
    and mode data extracted from W. C. Burton et al. The transport describes
    a right-angle junction: motion along x toward the origin, followed by
    motion along y away from the origin, with a symmetric (time-reversed)
    profile about the midpoint.

    Args:
        ion (IonSpecies, optional): Ion species used to define the mass
            and charge properties of the system. Defaults to CALCIUM_40.

    Returns:
        TransportProblem: the transport problem.
    """
    qx = _ramp(-350, 0, 0)
    qy = _ramp(0, 0, 350)
    qz = _mirror(data.yb_z_position, 350, 0)

    def waveform_index(t: Scalar) -> Scalar:
        return (speed / 700) * t

    def qbar(z: Scalar) -> Vector:
        return jnp.asarray([qx(z), qy(z), qz(z)]).flatten()

    theta = _hold_after_midpoint(data.crystal_angle, 350, 0)
    phi = _reverse_replay_after_midpoint(data.crystal_angle, 350, 0, sign=-1.0)

    return TransportProblem(
        ion=ion,
        waveform_index=waveform_index,
        qbar=qbar,
        freqs=(
            _mirror(data.mode_1, 350, 0),
            _mirror(data.mode_2, 350, 0),
            _mirror(data.mode_3, 350, 0),
        ),
        theta=theta,
        phi=phi,
        z_start=jnp.array(0.0),
        z_stop=jnp.array(1.0),
    )


def junction_constant_frequency_problem(
    speed: float = 50 * constants.meter / constants.second, ion: IonSpecies = CALCIUM_40
) -> TransportProblem:
    """Same as junction_problem, but with the frequencies held constant.

    Args:
        ion (IonSpecies, optional): Ion species used to define the mass
            and charge properties of the system. Defaults to CALCIUM_40.

    Returns:
        TransportProblem: the transport problem.
    """
    qx = _ramp(-350, 0, 0)
    qy = _ramp(0, 0, 350)
    qz = _mirror(data.yb_z_position, 350, 0)

    def waveform_index(t: Scalar) -> Scalar:
        return (speed / 700) * t

    def qbar(z: Scalar) -> Vector:
        return jnp.asarray([qx(z), qy(z), qz(z)]).flatten()

    theta = _hold_after_midpoint(data.crystal_angle, 350, 0)
    phi = _reverse_replay_after_midpoint(data.crystal_angle, 350, 0, sign=-1.0)

    return TransportProblem(
        ion=ion,
        waveform_index=waveform_index,
        qbar=qbar,
        freqs=(
            _constant(1.1216216216216217),
            _constant(1.8828828828828827),
            _constant(2.0),
        ),
        theta=theta,
        phi=phi,
        z_start=jnp.array(0.0),
        z_stop=jnp.array(1.0),
    )
