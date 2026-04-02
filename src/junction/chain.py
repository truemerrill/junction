import jax.numpy as jnp
from jaxtyping import Array, Float

from .species import IonSpecies


class IonChain:
    def __init__(self, *species: IonSpecies, ndim: int = 3):
        """Construct an ion chain

        Args:
            *species (IonSpecies): The ion species in the chain
            ndim (int, optional): The number of spatial dimensions. Defaults
                to 3.
        """
        self.species = tuple(species)
        self.ndim = ndim

    def __repr__(self) -> str:
        return "-".join([str(s) for s in self.species])

    @property
    def nion(self) -> int:
        return len(self.species)

    @property
    def M(self) -> Float[Array, "(N D) (N D)"]:
        """Calculate the mass matrix in a given unit system.

        Args:
            system (UnitSystem): the unit system.

        Returns:
            Array: the mass matrix.
        """
        Id = jnp.eye(self.ndim)
        m = jnp.array([s.mass for s in self.species])
        return jnp.kron(jnp.diag(m), Id)

    @property
    def Z(self) -> Float[Array, "(N D) D"]:
        """Calculate the charge matrix in a given unit system

        Args:
            system (UnitSystem): the unit system.

        Returns:
            Array: the charge matrix.
        """
        z = jnp.array([s.charge for s in self.species])
        Id = jnp.eye(self.ndim)
        Z = z[:, None, None] * Id
        return Z.reshape(-1, self.ndim)
