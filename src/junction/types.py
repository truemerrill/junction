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

ParameterFn: TypeAlias = Callable[[Scalar], Scalar]
"""A function which maps between the waveform index and some scalar parameter."""

VectorFn: TypeAlias = Callable[[Scalar], Vector]
"""A function which maps between the waveform index and a vector."""

ModeFreqs: TypeAlias = tuple[ParameterFn, ParameterFn, ParameterFn]
"""Tuple of functions for the mode frequencies of the three principle axes."""
