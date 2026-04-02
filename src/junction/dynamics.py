import jax
import jax.numpy as jnp

from .problem import TransportProblem
from .types import Matrix, Scalar

jax.config.update("jax_enable_x64", True)


def _angular_freq(freq: Scalar) -> Scalar:
    return 2 * jnp.pi * freq


def _rotate_y(angle: Scalar) -> Matrix:
    """Rotation matrix about the y-axis."""
    c = jnp.cos(angle * jnp.pi / 180)
    s = jnp.sin(angle * jnp.pi / 180)
    return jnp.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def _rotate_z(angle: Scalar) -> Matrix:
    """Rotation matrix about the z-axis."""
    c = jnp.cos(angle * jnp.pi / 180)
    s = jnp.sin(angle * jnp.pi / 180)
    return jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


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
    return (m**power) * jnp.eye(3)


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
    w = jnp.array([(_angular_freq(freq(z)) ** power) for freq in problem.freqs])
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
    M_minus_one_half = M(problem, -0.5)
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
