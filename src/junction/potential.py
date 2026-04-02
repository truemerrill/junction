import jax
import jax.numpy as jnp

from dataclasses import dataclass
from .species import IonSpecies
from typing import Callable, TypeAlias
from jaxtyping import Array, Float


jax.config.update("jax_enable_x64", True)


Scalar: TypeAlias = Float[Array, ""]
"""A jax-differentiable scalar"""

Vector: TypeAlias = Float[Array, "3"]
"""A 3-vector of reals"""

Matrix: TypeAlias = Float[Array, "3 3"]
"""A 3x3 matrix of reals"""

ParameterFn: TypeAlias = Callable[[Scalar], Scalar]
"""A function which maps between the waveform index and some scalar parameter."""

VectorFn: TypeAlias = Callable[[Scalar], Vector]
"""A function which maps between the waveform index and a vector."""

ModeFreqs: TypeAlias = tuple[ParameterFn, ParameterFn, ParameterFn]
"""Tuple of functions for the mode frequencies of the three principle axes."""


x = jnp.array([1, 0, 0])
y = jnp.array([0, 1, 0])
z = jnp.array([0, 0, 1])


def _angular_freq(freq: Scalar) -> Scalar:
    return 2 * jnp.pi * freq


def _rotate_y(angle: Scalar) -> Matrix:
    """Rotation matrix about the y-axis."""
    c = jnp.cos(angle * jnp.pi / 180)
    s = jnp.sin(angle * jnp.pi / 180)
    return jnp.array([
        [ c,  0.0,  s],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c],
    ])


def _rotate_z(angle: Scalar) -> Matrix:
    """Rotation matrix about the z-axis."""
    c = jnp.cos(angle * jnp.pi / 180)
    s = jnp.sin(angle * jnp.pi / 180)
    return jnp.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


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
            z axis (in degrees) between the axial mode and the x axis.
        phi (ParameterFn): function describing the rotation angle about the
            y axis (in degrees) between the axial mode and the x axis.
        z_start (Scalar): the starting waveform index.
        z_stop (Scalar): the stopping waveform index.
    """

    ion: IonSpecies
    qbar: VectorFn
    freqs: ModeFreqs
    theta: ParameterFn
    phi: ParameterFn
    z_start: Scalar
    z_stop: Scalar


def S(problem: TransportProblem, z: Scalar | None = None) -> Matrix:
    """Calculate the orthogonal matrix that diagonalizes the Hessian.

    Args:
        problem (TransportProblem): the transport problem.
        z (Scalar | None): the waveform index.  If not provided, we
            calculate the final rotation matrix at the end of the
            transport interval.

    Returns:
        Matrix: the orthogonal matrix
    """
    if z is None:
        z = problem.z_stop

    Ry = _rotate_y(problem.theta(z))
    Rz = _rotate_z(problem.phi(z))
    return Rz @ Ry


def M(problem: TransportProblem, power: float = 1.0) -> Matrix:
    """Calculate the mass matrix raised to some power.

    Note:
        The mass matrix is always diagonal    

    Args:
        problem (TransportProblem): the transport problem.
        power (float, optional): the power to raise the mass matrix to.
            Defaults to 1.0.

    Returns:
        Matrix: the mass matrix raised to the power
    """
    m = problem.ion.mass
    return (m ** power) * jnp.eye(3)


def Omega_tau(problem: TransportProblem, power: float = 1.0) -> Matrix:
    """Calculate the \\Omega_\\tau matrix raised to some power.

    Note:
        By definition, the \\Omega_\\tau matrix is always diagonal.

    Args:
        problem (TransportProblem): the transport problem.
        power (float, optional): the power to raise the Omega matrix
            to. Defaults to 1.0.

    Returns:
        Matrix: the \\Omega_\\tau matrix raised to the power
    """
    z = problem.z_stop
    w = jnp.array([
        (_angular_freq(freq(z)) ** power) for freq in problem.freqs
    ])
    return jnp.diag(w)


def K(problem: TransportProblem, z: Scalar) -> Matrix:
    """Calculate the Hessian of the potential 

    Args:
        problem (TransportProblem): the transport problem
        z (Scalar): the waveform index

    Returns:
        Matrix: the Hessian matrix
    """
    m = problem.ion.mass
    S_ = S(problem)

    k = jnp.array([m * _angular_freq(freq(z)) ** 2 for freq in problem.freqs])
    M_Omega2 = jnp.diag(k)

    return S_ @ M_Omega2 @ S_.T


def Omega2(problem: TransportProblem, z: Scalar) -> Matrix:
    """Calculate the Omega^2 matrix

    Note:
        This matrix is diagonal at the endpoint.        

    Args:
        problem (TransportProblem): the transport problem
        z (Scalar): the waveform index

    Returns:
        Matrix: the Omega^2 matrix
    """
    M_minus_one_half = M(problem, - 0.5)
    S_ = S(problem)
    K_ = K(problem, z)
    return S_.T @ M_minus_one_half @ K_ @ M_minus_one_half @ S_


def modes(problem: TransportProblem, z: Scalar, scale: float = 1.0) -> Matrix:
    """Construct scaled mode vectors in the lab frame for visualization.

    Note:
        This function returns a (3, 3) matrix whose columns are the three
        principal-axis mode vectors at waveform index ``z``.

    Args:
        problem (TransportProblem): the transport problem.
        z (Scalar): the waveform index.
        scale (float, optional): global scaling factor applied to all
            mode lengths. Defaults to 1.0.

    Returns:
        Matrix: a (3, 3) array whose columns are the three mode vectors
        in the lab frame.
    """
    v = jnp.array([scale * freq(z) for freq in problem.freqs])
    S_ = S(problem, z)
    return S_ @ jnp.diag(v)


# def modes(
#     freq_1: ParameterFn,
#     freq_2: ParameterFn,
#     freq_3: ParameterFn,
#     theta: ParameterFn,
#     phi: ParameterFn,
#     mass: float 
# ) -> tuple[
#         Callable[[Scalar], Matrix],
#         Callable[[Scalar], Matrix],
#     ]:
    
#     def eigenvalue(mass: float, freq: Scalar) -> Scalar:
#         omega = 2 * jnp.pi * freq
#         return mass * (omega ** 2)

#     def omega_squared(z: Scalar) -> Matrix:        
#         k = jnp.array([
#             eigenvalue(mass, freq_1(z)),
#             eigenvalue(mass, freq_2(z)),
#             eigenvalue(mass, freq_3(z))
#         ])
        
#         Omega2 = jnp.diag(k)
#         return Omega2
    
#     def potential_hessian(z: Scalar) -> Matrix:
#         Omega2 = omega_squared(z)
#         Ry = _rotate_y(theta(z))
#         Rz = _rotate_z(phi(z))
#         return Rz @ Ry @ Omega2 @ Ry.T @ Rz.T

#     return omega_squared, potential_hessian
