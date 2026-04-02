import jax
import jax.numpy as jnp
from jaxtyping import Array

from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize
from junction.yoshida4 import Yoshida4


def test_yoshida_forest_ruth_converges_4th_order_harmonic_oscillator():
    # Require float64 for tight tolerances / symplectic splitting accuracy.
    jax.config.update("jax_enable_x64", True)

    # Harmonic oscillator: q' = p/m, p' = -k q
    m = 1.0
    k = 2.0
    w = jnp.sqrt(k / m)

    def f_q(t, p, args): # type: ignore
        # dq/dt depends on p only
        return p / m

    def f_p(t, q, args): # type: ignore
        # dp/dt depends on q only
        return -k * q

    term_q = ODETerm(f_q)
    term_p = ODETerm(f_p)

    solver = Yoshida4()
    t0 = 0.0
    t1 = 3.0

    q0 = 0.7
    p0 = -0.2
    y0 = (jnp.array(q0), jnp.array(p0))

    def exact(t: float) -> tuple[Array, Array]:
        # q(t) = q0 cos(wt) + (p0/(m w)) sin(wt)
        # p(t) = p0 cos(wt) - (m w q0) sin(wt)
        ct = jnp.cos(w * t)
        st = jnp.sin(w * t)
        q = q0 * ct + (p0 / (m * w)) * st
        p = p0 * ct - (m * w * q0) * st
        return jnp.array(q), jnp.array(p)

    y_exact = exact(t1)

    def solve(dt0): # type: ignore
        sol = diffeqsolve(
            terms=(term_q, term_p),
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=SaveAt(t1=True),
            stepsize_controller=ConstantStepSize(),
            max_steps=int((t1 - t0) / dt0) + 10,
        )
        # saveat(t1=True) returns shape (1,) for each component
        qT = sol.ys[0][-1] # type: ignore
        pT = sol.ys[1][-1] # type: ignore
        return qT, pT

    dt = 1e-2
    y_dt = solve(dt)
    y_dt2 = solve(dt / 2)

    err_dt = jnp.sqrt((y_dt[0] - y_exact[0]) ** 2 + (y_dt[1] - y_exact[1]) ** 2)
    err_dt2 = jnp.sqrt((y_dt2[0] - y_exact[0]) ** 2 + (y_dt2[1] - y_exact[1]) ** 2)

    # 4th order: halving dt should reduce error by ~16x (asymptotically).
    # Use a loose bound to avoid flakiness across platforms.
    ratio = err_dt / err_dt2
    assert ratio > 8.0

    # Also ensure absolute accuracy is reasonable.
    assert err_dt2 < 1e-8
