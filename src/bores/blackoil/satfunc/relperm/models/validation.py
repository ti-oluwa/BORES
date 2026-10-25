"""Validation of the parameters shared by the analytical three-phase relative permeability models."""

from bores.errors import ValidationError
from bores.types import Number

__all__ = ["validate_analytical_relperm_parameters"]


def validate_analytical_relperm_parameters(
    *,
    irreducible_water_saturation: Number | None,
    residual_oil_saturation_water: Number | None,
    residual_oil_saturation_gas: Number | None,
    residual_gas_saturation: Number | None,
    maximum_water_relperm: Number,
    maximum_oil_relperm: Number,
    maximum_gas_relperm: Number,
    mixed_wet_water_fraction: Number,
    exponents: dict[str, Number] | None = None,
) -> None:
    """
    Check the parameters of an analytical three-phase relative permeability model.

    Only clearly invalid values are rejected. An endpoint relative permeability of zero would
    divide by zero when the two-phase curves are combined, an exponent that is not positive does
    not describe a relative permeability curve, and a water-wet fraction outside 0 to 1 weights
    the water-wet and oil-wet systems with a negative factor. Endpoint saturations are checked
    only when they are set on the model, because they can also be supplied for each evaluation.

    :param irreducible_water_saturation: Irreducible water saturation, or `None` if not set.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding, or `None`.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding, or `None`.
    :param residual_gas_saturation: Residual gas saturation, or `None`.
    :param maximum_water_relperm: Endpoint water relative permeability.
    :param maximum_oil_relperm: Endpoint oil relative permeability.
    :param maximum_gas_relperm: Endpoint gas relative permeability.
    :param mixed_wet_water_fraction: Fraction of the pore space that is water-wet in a mixed-wet rock.
    :param exponents: Corey exponents by name, for models that have them.
    :raises ValidationError: If a parameter is outside its valid range.
    """
    for name, saturation in (
        ("irreducible_water_saturation", irreducible_water_saturation),
        ("residual_oil_saturation_water", residual_oil_saturation_water),
        ("residual_oil_saturation_gas", residual_oil_saturation_gas),
        ("residual_gas_saturation", residual_gas_saturation),
    ):
        if saturation is not None and not 0.0 <= saturation <= 1.0:
            raise ValidationError(f"`{name}` must be between 0 and 1, got {saturation}.")

    for name, maximum in (
        ("maximum_water_relperm", maximum_water_relperm),
        ("maximum_oil_relperm", maximum_oil_relperm),
        ("maximum_gas_relperm", maximum_gas_relperm),
    ):
        if not 0.0 < maximum <= 1.0:
            raise ValidationError(f"`{name}` must be greater than 0 and at most 1, got {maximum}.")

    if not 0.0 <= mixed_wet_water_fraction <= 1.0:
        raise ValidationError(
            f"`mixed_wet_water_fraction` must be between 0 and 1, got {mixed_wet_water_fraction}."
        )

    for name, exponent in (exponents or {}).items():
        if not exponent > 0.0:
            raise ValidationError(f"`{name}` must be positive, got {exponent}.")
