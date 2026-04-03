import jax
import jax.numpy as jnp
from diffrax import ConstantStepSize, ODETerm, SaveAt, diffeqsolve

from .problem import TransportProblem
from .types import Array, Matrix3, Scalar
from .yoshida4 import Yoshida4

jax.config.update("jax_enable_x64", True)


__all__ = ("fundamental_matrix",)


def _angular_freq(freq: Scalar) -> Scalar:
    return 2 * jnp.pi * freq


def _rotate_y(angle: Scalar) -> Matrix3:
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


def _rotate_z(angle: Scalar) -> Matrix3:
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


def s_matrix(problem: TransportProblem, z: Scalar | None = None) -> Matrix3:
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


def m_matrix(problem: TransportProblem, power: float = 1.0) -> Matrix3:
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


def omega_tau_matrix(problem: TransportProblem, power: float = 1.0) -> Matrix3:
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


def hessian_matrix(
    problem: TransportProblem, z: Scalar, S: Matrix3 | None = None
) -> Matrix3:
    """Calculate the Hessian of the potential

    Args:
        problem (TransportProblem): the transport problem
        z (Scalar): the waveform index

    Returns:
        Matrix: the Hessian matrix
    """
    m = problem.ion.mass
    S_ = s_matrix(problem) if S is None else S
    k = jnp.array([m * _angular_freq(freq(z)) ** 2 for freq in problem.freqs])
    M_Omega2 = jnp.diag(k)

    return S_ @ M_Omega2 @ S_.T


def omega_squared_matrix(
    problem: TransportProblem,
    z: Scalar,
    S: Matrix3 | None = None,
    M_minus_one_half: Matrix3 | None = None,
) -> Matrix3:
    """Calculate the Omega^2 matrix

    Note:
        This matrix is diagonal at the endpoint.

    Args:
        problem (TransportProblem): the transport problem
        z (Scalar): the waveform index

    Returns:
        Matrix: the Omega^2 matrix
    """
    S_ = s_matrix(problem) if S is None else S
    M_minus_one_half_ = (
        m_matrix(problem, -0.5) if M_minus_one_half is None else M_minus_one_half
    )
    K_ = hessian_matrix(problem, z, S)
    return S_.T @ M_minus_one_half_ @ K_ @ M_minus_one_half_ @ S_


def fundamental_matrix(
    problem: TransportProblem, time: Array, dt: float | None = None
) -> Array:
    """Solve for the fundamental matrix U(t) on a time grid.

    This solves

        dU/dt = A(t) U(t),    U(t0) = I

    where U(t) is the principal fundamental matrix for the linear system.
    A symplectic ODE solver is used.

    Args:
        problem (TransportProblem): The transport problem.
        time (Array): Monotone array of time points at which to save U(t).
            These are the time samples which we use to compute the convolution
            integral.
        dt (float | None, optional): Initial step size. If not provided,
            uses the spacing of the first two time points.

    Returns:
        Array: Array of shape (nt, 6, 6) containing U(t) evaluated at each
            requested time point.
    """
    if len(time) < 2:
        raise ValueError("Must solve at at least two time points")

    t0 = time[0]
    t1 = time[-1]
    dt0 = time[1] - time[0] if dt is None else dt

    # Pre-compute all of the constant matrices we need. This saves us from
    # having to recompute each of these.  JAX breaks if we use an lru_cache,
    # so it's better to essentially cache by hand in case we want to take
    # gradients in the future ...

    omega_tau = omega_tau_matrix(problem)
    omega_tau_minus_one_half = omega_tau_matrix(problem, power=-0.5)
    S = s_matrix(problem)
    M_minus_one_half = m_matrix(problem, power=-0.5)
    I = jnp.eye(3, dtype=jnp.float64)
    Z = jnp.zeros((3, 3), dtype=jnp.float64)

    def omega_tilde(t: Scalar) -> Matrix3:
        z = problem.waveform_index(t)
        omega_squared = omega_squared_matrix(problem, z, S, M_minus_one_half)
        return omega_tau_minus_one_half @ omega_squared @ omega_tau_minus_one_half

    # Construct the terms for the symplectic solver
    def f_q(
        t: Scalar, P: tuple[Matrix3, Matrix3], args: None
    ) -> tuple[Matrix3, Matrix3]:
        Upq, Upp = P
        return (omega_tau @ Upq, omega_tau @ Upp)

    def f_p(
        t: Scalar, Q: tuple[Matrix3, Matrix3], args: None
    ) -> tuple[Matrix3, Matrix3]:
        Uqq, Uqp = Q
        Ot = omega_tilde(t)
        return (-Ot @ Uqq, -Ot @ Uqp)

    term_q = ODETerm(f_q)  # type: ignore
    term_p = ODETerm(f_p)  # type: ignore

    # Set the initial conditions.  At the start, the fundamental matrix is the
    # identity

    Q0 = (I, Z)
    P0 = (Z, I)
    y0 = (Q0, P0)

    # Solve the ODE
    sol = diffeqsolve(
        terms=(term_q, term_p),
        solver=Yoshida4(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=SaveAt(ts=time),
        stepsize_controller=ConstantStepSize(),
    )

    if sol.ys is None:
        raise ValueError("Failed to solve ODE")

    # Restructure the symplectic block structure into the standard matrix form
    (Uqq, Uqp), (Upq, Upp) = sol.ys
    top = jnp.concatenate([Uqq, Uqp], axis=-1)  # (N, 3, 6)
    bottom = jnp.concatenate([Upq, Upp], axis=-1)  # (N, 3, 6)
    U = jnp.concatenate([top, bottom], axis=-2)  # (N, 6, 6)

    return U


def b_matrix() -> Array:
    return jnp.kron(
        jnp.array([[0], [1]], dtype=jnp.float64), jnp.eye(3, dtype=jnp.float64)
    )


def modes(problem: TransportProblem, z: Scalar, scale: float = 1.0) -> Matrix3:
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
    S_ = s_matrix(problem, z)
    return S_ @ jnp.diag(v)
