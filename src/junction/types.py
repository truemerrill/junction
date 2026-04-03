from typing import Callable, TypeAlias

import jax
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)


Scalar: TypeAlias = Float[Array, ""]
"""A jax-differentiable scalar"""

Vector: TypeAlias = Float[Array, "3"]
"""A 3-vector of reals"""

Matrix3: TypeAlias = Float[Array, "3 3"]
"""A 3x3 matrix of reals"""

Matrix6: TypeAlias = Float[Array, "6 6"]
"""A 6x6 matrix of reals"""

TimeArray: TypeAlias = Float[Array, "N"]  # noqa: F821

Matrix6Array: TypeAlias = Float[Array, "N 6 6"]

ParameterFn: TypeAlias = Callable[[Scalar], Scalar]
"""A function which maps between the waveform index and some scalar parameter."""

VectorFn: TypeAlias = Callable[[Scalar], Vector]
"""A function which maps between the waveform index and a vector."""

ColoredNoiseFn: TypeAlias = Callable[[Scalar, Scalar], Matrix3]
"""A function of the form G(s, s') -> M, where M is a 3x3 matrix."""

WhiteNoiseFn: TypeAlias = Callable[[Scalar], Matrix3]
"""A function of the form G(s) -> M, were M is a 3x3 matrix."""

ModeFreqs: TypeAlias = tuple[ParameterFn, ParameterFn, ParameterFn]
"""Tuple of functions for the mode frequencies of the three principle axes."""
