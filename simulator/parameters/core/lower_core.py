#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from dataclasses import dataclass, field

from simulator.parameters.base import RegisteredEnum
from simulator.parameters.core.core import (
    CoreParameters,
    CoreMappingParameters,
)


class BalancedCoreStyle(RegisteredEnum):
    """What style of balanced core to use.

    "ONE_SIDED" : One of the devices is always at the lowest conductance
    "TWO_SIDED" : The sum of the two conductances is fixed
    """

    # TODO: Better name? Signed core style?
    ONE_SIDED = 1
    TWO_SIDED = 2


def get_signed_core_mapping_defaults() -> CoreMappingParameters:
    """Returns the default mapping of signed cores."""
    SignedCoreMappingDefaults = CoreMappingParameters(
        weights={"min": -1.0, "max": 1.0, "percentile": None},
    )
    return SignedCoreMappingDefaults


def get_unsigned_core_mapping_defaults() -> CoreMappingParameters:
    """Returns the default mapping of unsigned cores."""
    UnsignedCoreMappingDefaults = CoreMappingParameters(
        weights={"min": 0.0, "max": 1.0, "percentile": None},
    )
    return UnsignedCoreMappingDefaults


@dataclass(repr=False)
class SignedCoreParameters(CoreParameters):
    """Parameters for describing signed core behavior.

    Args:
        core_type: Type of core associated with the core parameter
        mapping: Mapping parameters for the core.
        subcores: Contains parameters for child cores.
        adc_scheme: Scheme used to describe ADC initialization.
        dac_scheme: Scheme used to describe DAC initialization.
        style: Style to use for conductance matrix mapping
        interleaved_posneg: If true, interleave positive and negative matrices
        subtract_in_xbar: If true, results from matrix multiplications on
            positive and negative xbars will be subtracted in the crossbar
            before the ADC.
    """

    core_type: str = "SignedCore"
    mapping: CoreMappingParameters = field(
        default_factory=get_signed_core_mapping_defaults,
    )
    interleaved_posneg: bool = False
    subtract_in_xbar: bool = False
    style: BalancedCoreStyle = BalancedCoreStyle.ONE_SIDED

    def validate(self) -> None:
        """Validates the configuration of the signed core parameters.

        Raises:
            ValueError: Raised if an invalid configuration is provided.
        """
        if self.interleaved_posneg and not self.subtract_in_xbar:
            raise ValueError("interleave_posneg requires subtract_in_xbar")
        return super().validate()


@dataclass(repr=False)
class UnsignedCoreParameters(CoreParameters):
    """Parameters for describing signed core behavior.

    Args:
        core_type: Type of core associated with the core parameter
        mapping: Mapping parameters for the core.
        subcores: Contains parameters for child cores.
        adc_scheme: Scheme used to describe ADC initialization.
        dac_scheme: Scheme used to describe DAC initialization.
    """

    core_type: str = "UnsignedCore"
    mapping: CoreMappingParameters = field(
        default_factory=get_unsigned_core_mapping_defaults,
    )

    def validate(self) -> None:
        """Validates the configuration of the parameter."""
        if self.mapping.weights.min != 0:
            raise ValueError("Minimum weight value in unsigned core must be 0")
        return super().validate()
