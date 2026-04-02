from dataclasses import dataclass

from .unit import constants



@dataclass(frozen=True)
class IonSpecies:
    name: str
    mass: float
    charge: float

    def __repr__(self) -> str:
        return self.name


BERYLLIUM_9 = IonSpecies(
    name="⁹Be⁺",
    mass=9 * constants.amu,
    charge=1 * constants.e_charge,
)

MAGNESIUM_24 = IonSpecies(
    name="²⁴Mg⁺",
    mass=24 * constants.amu,
    charge=1 * constants.e_charge,
)

CALCIUM_40 = IonSpecies(
    name="⁴⁰Ca⁺", mass=40 * constants.amu, charge=1 * constants.e_charge
)

CALCIUM_43 = IonSpecies(
    name="⁴³Ca⁺",
    mass=43 * constants.amu,
    charge=1 * constants.e_charge,
)

STRONTIUM_88 = IonSpecies(
    name="⁸⁸Sr⁺",
    mass=88 * constants.amu,
    charge=1 * constants.e_charge,
)

BARIUM_137 = IonSpecies(
    name="¹³⁷Ba⁺",
    mass=137 * constants.amu,
    charge=1 * constants.e_charge,
)

BARIUM_138 = IonSpecies(
    name="¹³⁸Ba⁺",
    mass=138 * constants.amu,
    charge=1 * constants.e_charge,
)

YTTERBIUM_171 = IonSpecies(
    name="¹⁷¹Yb⁺",
    mass=171 * constants.amu,
    charge=1 * constants.e_charge,
)

YTTERBIUM_174 = IonSpecies(
    name="¹⁷⁴Yb⁺",
    mass=174 * constants.amu,
    charge=1 * constants.e_charge,
)
