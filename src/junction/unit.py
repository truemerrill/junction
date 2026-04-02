import math
from dataclasses import dataclass
from typing import Protocol, Self, TypeVar


@dataclass(frozen=True)
class UnitSystem:
    """Dataclass that sets the scales of the computational unit system.

    Note:
        Let `x` represent a quantity in the computational unit system, and let
        `X` represent the same quantity in SI units.  The attributes of this
        class are precisely the conversion factors `g` such that `X = g x`.

        For example, suppose the computational system measured length in units
        of microns.  Then `1e-6 x = X` and `g = 1e-6`, which is the length of
        a single micron measured in meters.
    """

    length: float = 1.0
    mass: float = 1.0
    time: float = 1.0
    charge: float = 1.0
    temperature: float = 1.0

    @property
    def area(self) -> float:
        return self.length * self.length

    @property
    def volume(self) -> float:
        return self.area * self.length

    @property
    def frequency(self) -> float:
        return 1 / self.time

    @property
    def velocity(self) -> float:
        return self.length / self.time

    @property
    def acceleration(self) -> float:
        return self.velocity / self.time

    @property
    def momentum(self) -> float:
        return self.mass * self.velocity

    @property
    def force(self) -> float:
        return self.mass * self.acceleration

    @property
    def pressure(self) -> float:
        return self.force / self.area

    @property
    def energy(self) -> float:
        return self.force * self.length

    @property
    def power(self) -> float:
        return self.energy / self.time

    @property
    def action(self) -> float:
        return self.energy * self.time

    @property
    def current(self) -> float:
        return self.charge / self.time

    @property
    def voltage(self) -> float:
        return self.energy / self.charge

    @property
    def electric_field(self) -> float:
        return self.force / self.charge

    @property
    def magnetic_field(self) -> float:
        return self.force / (self.charge * self.velocity)

    @property
    def capacitance(self) -> float:
        return self.charge / self.voltage

    @property
    def resistance(self) -> float:
        return self.voltage / self.current

    @property
    def impedance(self) -> float:
        return self.resistance

    @property
    def conductance(self) -> float:
        return self.current / self.voltage

    @property
    def permittivity(self) -> float:
        return self.capacitance / self.length

    @property
    def permeability(self) -> float:
        return self.mass * self.length / self.charge**2

    @property
    def entropy(self) -> float:
        return self.energy / self.temperature


SI = UnitSystem(length=1.0, mass=1.0, time=1.0, charge=1.0, temperature=1.0)
CGS = UnitSystem(
    length=1e-2,  # cm
    mass=1e-3,  # g
    time=1.0,  # s
    charge=1.0,  # C
    temperature=1.0,  # K
)
ION_TRAP = UnitSystem(
    length=1e-6,  # micron
    mass=1.66053906660e-27,  # AMU
    time=1e-6,  # microsecond
    charge=1.602176634e-19,  # elementary charge
    temperature=1.0,  # Kelvin
)


dimension: UnitSystem = ION_TRAP
"""Global constant storing the computational length scales"""


def set_unit_system(system: UnitSystem) -> UnitSystem:
    """Set the global computational unit system.

    Args:
        system (Scale): set of unit scales that define the unit system.

    Returns:
        Scale: the prior set of scales.
    """
    global dimension
    old_system = dimension
    dimension = system
    return old_system


class SupportsScale(Protocol):
    """Protocol for any type that can be multiplied or divided by a float."""

    def __mul__(self, other: float, /) -> Self: ...
    def __truediv__(self, other: float, /) -> Self: ...


V = TypeVar("V", bound=SupportsScale)


def to_comp(si_value: V, scale: float) -> V:
    """Convert from an SI value to a value in the computational system.

    Args:
        si_value (V): the SI value.
        scale (float): the dimension scale.

    Returns:
        V: the value in computational units
    """
    return si_value / scale


def to_si(comp_value: V, scale: float) -> V:
    """Convert from computational units to SI units.

    Args:
        comp_value (V): the value in computational units.
        scale (float): the dimension scale.

    Returns:
        V: the value in SI units.
    """
    return comp_value * scale


class Constants:
    @property
    def hbar(self) -> float:
        """Reduced Planck's constant.

        Returns:
            float: reduced Planck's constant in the computational unit system.
        """
        return self.h / (2 * math.pi)

    @property
    def h(self) -> float:
        """Planck's constant.

        Returns:
            float: Planck's constant in the computational unit system.
        """
        return to_comp(6.62607015e-34, dimension.action)

    @property
    def c(self) -> float:
        """Speed of light.

        Returns:
            float: the speed of light in the computational unit system.
        """
        return to_comp(299792458.0, dimension.velocity)

    @property
    def MHz(self) -> float:
        """Megahertz.

        Returns:
            float: a megahertz in the computational unit system.
        """
        return to_comp(1e6, dimension.frequency)

    @property
    def second(self) -> float:
        """Second.

        Returns:
            float: a second in the computational unit system.
        """
        return to_comp(1.0, dimension.time)

    @property
    def s(self) -> float:
        """Second.

        Returns:
            float: a second in the computational unit system.
        """
        return self.second

    @property
    def microsecond(self) -> float:
        """Microsecond

        Returns:
            float: a microsecond in the computational unit system
        """
        return to_comp(1e-6, dimension.time)

    @property
    def µs(self) -> float:
        """Microsecond

        Returns:
            float: a microsecond in teh computational unit system
        """
        return self.microsecond

    @property
    def nanosecond(self) -> float:
        """Nanosecond

        Returns:
            float: a nanosecond in the computational unit system
        """
        return to_comp(1e-9, dimension.time)

    @property
    def ns(self) -> float:
        """Nanosecond

        Returns:
            float: a nanosecond in the computational unit system
        """
        return self.nanosecond

    @property
    def volt(self) -> float:
        """Volt

        Returns:
            float: a volt in the computational unit system
        """
        return to_comp(1.0, dimension.voltage)

    @property
    def v(self) -> float:
        """Volt

        Returns:
            float: a volt in the computational unit system
        """
        return self.volt

    @property
    def volt_per_meter(self) -> float:
        """Volt per meter

        Returns:
            float: a volt per meter in the computational unit system
        """
        return to_comp(1.0, dimension.electric_field)

    @property
    def volt_per_meter_squared(self) -> float:
        """Volt per meter squared

        Returns:
            float: a volt per meter squared in the computational unit system
        """
        return self.volt / (self.meter * self.meter)

    @property
    def meter(self) -> float:
        """Meter

        Returns:
            float: a meter in the computational unit system
        """
        return to_comp(1.0, dimension.length)

    @property
    def m(self) -> float:
        """Meter

        Returns:
            float: a meter in the computational unit system
        """
        return self.meter

    @property
    def micron(self) -> float:
        """Micron

        Returns:
            float: a micron in the computational unit system
        """
        return to_comp(1e-6, dimension.length)

    @property
    def µm(self) -> float:
        """Micron

        Returns:
            float: a micron in the computational unit system
        """
        return self.micron

    @property
    def e_charge(self) -> float:
        """The fundamental electric charge

        Returns:
            float: the fundamental charge in the computational unit system
        """
        return to_comp(1.602176634e-19, dimension.charge)

    @property
    def boltzmann_constant(self) -> float:
        """Boltzmann's constant

        Returns:
            float: the Boltzmann constant in the computational unit system
        """
        return to_comp(1.380649e-23, dimension.entropy)

    @property
    def k_b(self) -> float:
        """Boltzmann's constant

        Returns:
            float: the Boltzmann constant in the computational unit system
        """
        return self.boltzmann_constant

    @property
    def amu(self) -> float:
        """Atomic mass unit

        Returns:
            float: an AMU in the computational unit system
        """
        return to_comp(1.66053906892e-27, dimension.mass)

    @property
    def mass_electron(self) -> float:
        """The electron mass

        Returns:
            float: the electron mass in the computational unit system
        """
        return to_comp(9.1093837139e-31, dimension.mass)

    @property
    def m_e(self) -> float:
        """The electron mass

        Returns:
            float: the electron mass in the computational unit system
        """
        return self.mass_electron

    @property
    def mass_proton(self) -> float:
        """The proton mass

        Returns:
            float: the proton mass in the computational unit system
        """
        return to_comp(1.67262192595e-27, dimension.mass)

    @property
    def m_p(self) -> float:
        """The proton mass

        Returns:
            float: the proton mass in the computational unit system
        """
        return self.mass_proton

    @property
    def mass_neutron(self) -> float:
        """The neutron mass

        Returns:
            float: the neutron mass in the computational unit system
        """
        return to_comp(1.67492750056e-27, dimension.mass)

    @property
    def m_n(self) -> float:
        """The neutron mass

        Returns:
            float: the neutron mass in the computational unit system
        """
        return self.mass_neutron

    @property
    def vacuum_permittivity(self) -> float:
        """The permittivity of free space

        Returns:
            float: the vacuum permittivity in the computational unit system
        """
        return to_comp(8.8541878188e-12, dimension.permittivity)

    @property
    def epsilon_0(self) -> float:
        """The permittivity of free space

        Returns:
            float: the vacuum permittivity in the computational unit system
        """
        return self.vacuum_permittivity

    @property
    def vacuum_permeability(self) -> float:
        """The permeability of free space

        Returns:
            float: the vacuum permeability in the computational unit system
        """
        return to_comp(1.25663706127e-6, dimension.permeability)

    @property
    def mu_0(self) -> float:
        """The permeability of free space

        Returns:
            float: the vacuum permeability in the computational unit system
        """
        return self.vacuum_permeability

    @property
    def coulomb_constant(self) -> float:
        """The Coulomb constant

        Returns:
            float: the Coulomb constant in the computational unit system
        """
        return 1 / (4 * math.pi * self.vacuum_permittivity)


constants = Constants()
