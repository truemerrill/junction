# pyright: reportMissingTypeArgument=false

from typing import Callable, ClassVar, TypeAlias

from diffrax import AbstractSolver, AbstractTerm
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]
Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]


class Yoshida4(AbstractSolver):
    """Yoshida / Forest-Ruth 4th-order symmetric split solver.

    Advances systems of the form

        y_a' = f_a(t, y_b)
        y_b' = f_b(t, y_a)

    by composing a symmetric second-order Strang kernel with Yoshida's
    fourth-order coefficients.

    For time-dependent split terms, each partial update is evaluated at the
    midpoint time of its own subinterval. This is the critical change needed
    for the method to remain symmetric when either term depends explicitly
    on time.
    """

    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms: PyTree[AbstractTerm]) -> int | None:
        return 4

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        term_a, term_b = terms
        ya, yb = y0
        dt = t1 - t0

        two_to_third: float = 2.0 ** (1.0 / 3.0)
        denom = 2.0 - two_to_third
        w1 = 1.0 / denom
        w0 = -two_to_third / denom

        def a_step(t_start: RealScalarLike, h: RealScalarLike, ya: Ya, yb: Yb) -> Ya:
            """Advance ya over the subinterval [t_start, t_start + h]."""
            t_mid = t_start + 0.5 * h
            control = term_a.contr(t_start, t_start + h)
            delta = term_a.vf_prod(t_mid, yb, args, control)
            return (ya**ω + delta**ω).ω

        def b_step(t_start: RealScalarLike, h: RealScalarLike, ya: Ya, yb: Yb) -> Yb:
            """Advance yb over the subinterval [t_start, t_start + h]."""
            t_mid = t_start + 0.5 * h
            control = term_b.contr(t_start, t_start + h)
            delta = term_b.vf_prod(t_mid, ya, args, control)
            return (yb**ω + delta**ω).ω

        def strang_step(
            t_start: RealScalarLike, h: RealScalarLike, ya: Ya, yb: Yb
        ) -> tuple[Ya, Yb]:
            """Symmetric second-order Strang step.

            S2(h) = A(h/2) -> B(h) -> A(h/2)
            """
            hh = 0.5 * h

            # First A half-step over [t_start, t_start + hh]
            ya = a_step(t_start, hh, ya, yb)

            # Full B step over [t_start, t_start + h]
            yb = b_step(t_start, h, ya, yb)

            # Final A half-step over [t_start + hh, t_start + h]
            ya = a_step(t_start + hh, hh, ya, yb)

            return ya, yb

        t = t0

        h = w1 * dt
        ya, yb = strang_step(t, h, ya, yb)
        t = t + h

        h = w0 * dt
        ya, yb = strang_step(t, h, ya, yb)
        t = t + h

        h = w1 * dt
        ya, yb = strang_step(t, h, ya, yb)

        y_out = (ya, yb)
        dense_info = dict(y0=y0, y1=y_out)
        return y_out, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        term_a, term_b = terms
        ya, yb = y0
        fa = term_a.vf(t0, yb, args)
        fb = term_b.vf(t0, ya, args)
        return fa, fb


# from typing import Callable, ClassVar, TypeAlias

# from diffrax import AbstractSolver, AbstractTerm
# from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
# from diffrax._local_interpolation import LocalLinearInterpolation
# from diffrax._solution import RESULTS
# from equinox.internal import ω
# from jaxtyping import ArrayLike, Float, PyTree

# _ErrorEstimate: TypeAlias = None
# _SolverState: TypeAlias = None

# Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]
# Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]


# class Yoshida4(AbstractSolver):
#     """Yoshida / Forest-Ruth 4th order symplectic method."""

#     term_structure: ClassVar = (AbstractTerm, AbstractTerm)
#     interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
#         LocalLinearInterpolation
#     )

#     def order(self, terms: PyTree[AbstractTerm]) -> int | None:
#         return 4

#     def init(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#     ) -> _SolverState:
#         return None

#     def step(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#         solver_state: _SolverState,
#         made_jump: BoolScalarLike,
#     ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
#         del solver_state, made_jump

#         term_1, term_2 = terms
#         y1, y2 = y0
#         dt = t1 - t0

#         # Forest–Ruth / Yoshida coefficients (4th order via 3x Strang composition)
#         #
#         # w1 = 1 / (2 - 2^(1/3)),
#         # w0 = -2^(1/3) / (2 - 2^(1/3))

#         two_to_third: float = 2.0 ** (1.0 / 3.0)
#         denom = 2.0 - two_to_third
#         w1 = 1.0 / denom
#         w0 = -two_to_third / denom

#         def A_step(
#             t_start: RealScalarLike, h: RealScalarLike, y1: Ya, y2: Yb
#         ) -> tuple[Ya, Yb]:
#             # y1 <- y1 + h * f1(t, y2)
#             control1 = term_1.contr(t_start, t_start + h)
#             return (y1**ω + term_1.vf_prod(t_start, y2, args, control1) ** ω).ω

#         def B_step(
#             t_start: RealScalarLike, h: RealScalarLike, y1: Ya, y2: Yb
#         ) -> tuple[Ya, Yb]:
#             # y2 <- y2 + h * f2(t, y1)
#             control2 = term_2.contr(t_start, t_start + h)
#             return (y2**ω + term_2.vf_prod(t_start, y1, args, control2) ** ω).ω

#         def strang_step(
#             t_start: RealScalarLike, h: RealScalarLike, y1: Ya, y2: Yb
#         ) -> tuple[Ya, Yb]:
#             # 2nd order: A(h/2) -> B(h) -> A(h/2)
#             hh = 0.5 * h
#             y1 = A_step(t_start, hh, y1, y2)
#             y2 = B_step(t_start + hh, h, y1, y2)
#             y1 = A_step(t_start + hh, hh, y1, y2)
#             return y1, y2

#         # Compose 2nd-order Strang to get 4th-order Yoshida / Forest–Ruth:
#         # S4(dt) = S2(w1*dt) ∘ S2(w0*dt) ∘ S2(w1*dt)
#         t = t0
#         h = w1 * dt
#         y1, y2 = strang_step(t, h, y1, y2)
#         t = t + h

#         h = w0 * dt
#         y1, y2 = strang_step(t, h, y1, y2)
#         t = t + h

#         h = w1 * dt
#         y1, y2 = strang_step(t, h, y1, y2)
#         y_out = (y1, y2)
#         dense_info = dict(y0=y0, y1=y_out)
#         return y_out, None, dense_info, None, RESULTS.successful

#     def func(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#     ) -> VF:
#         term_1, term_2 = terms
#         y0_1, y0_2 = y0
#         f1 = term_1.vf(t0, y0_2, args)
#         f2 = term_2.vf(t0, y0_1, args)
#         return f1, f2


# # pyright: reportMissingTypeArgument=false

# from typing import Callable, ClassVar, TypeAlias

# from diffrax import AbstractSolver, AbstractTerm
# from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
# from diffrax._local_interpolation import LocalLinearInterpolation
# from diffrax._solution import RESULTS
# from equinox.internal import ω
# from jaxtyping import ArrayLike, Float, PyTree

# _ErrorEstimate: TypeAlias = None
# _SolverState: TypeAlias = None

# Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]
# Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]


# class Yoshida4(AbstractSolver):
#     """Yoshida / Forest-Ruth 4th-order symmetric split solver.

#     Advances systems of the form

#         y_a' = f_a(t, y_b)
#         y_b' = f_b(t, y_a)

#     by composing a symmetric second-order Strang kernel with Yoshida's
#     fourth-order coefficients.

#     For time-dependent split terms, each partial update is evaluated at the
#     midpoint time of its own subinterval. This is the critical change needed
#     for the method to remain symmetric when either term depends explicitly
#     on time.
#     """

#     term_structure: ClassVar = (AbstractTerm, AbstractTerm)
#     interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
#         LocalLinearInterpolation
#     )

#     def order(self, terms: PyTree[AbstractTerm]) -> int | None:
#         return 4

#     def init(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#     ) -> _SolverState:
#         del terms, t0, t1, y0, args
#         return None

#     def step(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#         solver_state: _SolverState,
#         made_jump: BoolScalarLike,
#     ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
#         del solver_state, made_jump

#         term_a, term_b = terms
#         ya, yb = y0
#         dt = t1 - t0

#         two_to_third: float = 2.0 ** (1.0 / 3.0)
#         denom = 2.0 - two_to_third
#         w1 = 1.0 / denom
#         w0 = -two_to_third / denom

#         def a_step(
#             t_start: RealScalarLike, h: RealScalarLike, ya: Ya, yb: Yb
#         ) -> Ya:
#             """Advance ya over the subinterval [t_start, t_start + h]."""
#             t_mid = t_start + 0.5 * h
#             control = term_a.contr(t_start, t_start + h)
#             delta = term_a.vf_prod(t_mid, yb, args, control)
#             return (ya**ω + delta**ω).ω

#         def b_step(
#             t_start: RealScalarLike, h: RealScalarLike, ya: Ya, yb: Yb
#         ) -> Yb:
#             """Advance yb over the subinterval [t_start, t_start + h]."""
#             t_mid = t_start + 0.5 * h
#             control = term_b.contr(t_start, t_start + h)
#             delta = term_b.vf_prod(t_mid, ya, args, control)
#             return (yb**ω + delta**ω).ω

#         def strang_step(
#             t_start: RealScalarLike, h: RealScalarLike, ya: Ya, yb: Yb
#         ) -> tuple[Ya, Yb]:
#             """Symmetric second-order Strang step.

#             S2(h) = A(h/2) -> B(h) -> A(h/2)
#             """
#             hh = 0.5 * h

#             # First A half-step over [t_start, t_start + hh]
#             ya = a_step(t_start, hh, ya, yb)

#             # Full B step over [t_start, t_start + h]
#             yb = b_step(t_start, h, ya, yb)

#             # Final A half-step over [t_start + hh, t_start + h]
#             ya = a_step(t_start + hh, hh, ya, yb)

#             return ya, yb

#         t = t0

#         h = w1 * dt
#         ya, yb = strang_step(t, h, ya, yb)
#         t = t + h

#         h = w0 * dt
#         ya, yb = strang_step(t, h, ya, yb)
#         t = t + h

#         h = w1 * dt
#         ya, yb = strang_step(t, h, ya, yb)

#         y_out = (ya, yb)
#         dense_info = dict(y0=y0, y1=y_out)
#         return y_out, None, dense_info, None, RESULTS.successful

#     def func(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#     ) -> VF:
#         term_a, term_b = terms
#         ya, yb = y0
#         fa = term_a.vf(t0, yb, args)
#         fb = term_b.vf(t0, ya, args)
