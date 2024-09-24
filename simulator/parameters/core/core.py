#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import re
import logging

from typing import Any
from dataclasses import dataclass

from simulator.backend.registry import register_subclasses, RegistryManager
from simulator.parameters.base import (
    BasePairedParameters,
    BaseParameters,
    RegisteredEnum,
)

log = logging.getLogger(__name__)


class ConvertingConfiguration(RegisteredEnum):
    """Describes the scheme for behavior of the ADC and DAC of a core.

    "SKIP_CONVERSION": ADC/DAC ownership is not defined at the current
        level of the core
    "SHARED_PER_CHILD": ADC/DAC is shared between all child cores.
    "UNIQUE_PER_CHILD": ADC/DAC is unique between all child cores.
    """

    SKIP_CONVERSION = 0
    SHARED_PER_CHILD = 1
    UNIQUE_PER_CHILD = 2


# TODO: Can we simplify this because core types overlap?
class CoreStyle(RegisteredEnum):
    """What style of core to use.

    "BALANCED" TODO: Description
    "BITSLICED"  TODO: Description
    "OFFSET"  TODO: Description
    """

    BALANCED = 1
    BITSLICED = 2
    OFFSET = 3


@register_subclasses
@dataclass(repr=False)
class CoreParameters(BaseParameters):
    """Describes the configuration of a generic core.

    Args:
        core_type: Type of core the parameter is for.
        mapping: Mapping parameters for the core.
        subcores: Contains parameters for child cores.
        adc_scheme: Scheme used to describe ADC initialization.
        dac_scheme: Scheme used to describe DAC initialization.
    """

    core_type: str = "ICore"
    mapping: CoreMappingParameters = None
    subcores: dict[Any, CoreParameters] = None
    adc_scheme: ConvertingConfiguration = ConvertingConfiguration.SKIP_CONVERSION
    dac_scheme: ConvertingConfiguration = ConvertingConfiguration.SKIP_CONVERSION
    weight_bits: int = 0
    # TODO: Document this!

    def __new__(cls, *args, **kwargs):
        """Returns an unintialized instance of the class."""
        registry_manager = RegistryManager()
        key_name = "core_type"
        key_value = kwargs.get("core_type", cls.core_type)
        param_class = registry_manager.get_from_key(
            parent=CoreParameters,
            key_name=key_name,
            key_value=key_value,
        )
        param = super().__new__(param_class)
        return param

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute on the object.

        For CoreParameters, a hook is added to subcores keyword to convert
        dict values to CoreParameters.
        """
        if name == "subcores" and value is not None:
            # Automatically cast dicts to appropriate core param type
            for k, v in value.items():
                if issubclass(type(v), CoreParameters):
                    v._parent = self
                    continue
                try:
                    registry_manager = RegistryManager()
                    key_name = "core_type"
                    key_value = v.get(key_name)
                    param_class = registry_manager.get_from_key(
                        parent=CoreParameters,
                        key_name=key_name,
                        key_value=key_value,
                    )
                    value[k] = param_class(**v)
                    value[k]._parent = self
                except KeyError as e:
                    core_type = v.get("core_type")
                    msg = f"""Cannot find core type '{core_type}'. Was it imported?"""
                    raise KeyError(msg) from e
        return super().__setattr__(name, value)

    def __getitem__(self, name: str) -> Any:
        """Gets an item from the object.

        For CoreParameters, adds a hook to support getting nested items from the
        subcores dictionary.
        """
        if "." not in name:
            return super().__getitem__(name)
        name, _, subname = name.partition(".")
        if name == "subcores":
            # Add a hook on getitem so that it can grab appropriate
            # subcores (which are dicts, not params)
            if "." not in subname:
                return super().__getitem__(name)[int(subname)]
            subcore_name, _, subname = subname.partition(".")
            return getattr(self, name)[int(subcore_name)][subname]
        return super().__getitem__(name)


@dataclass(repr=False)
class MappingParameters(BaseParameters):
    """Parameters for mapping outputs.

    Attributes:
        clipping: Whether to use clipping or not
        min: Minimum value before clipping
        max: Maximum value before clipping
        percentile: Percentile to clip at, if used
    """

    clipping: bool = False
    min: float = -1.0
    max: float = 1.0
    percentile: float = None

    @property
    def range(self) -> float:
        """Range of the mapping."""
        return self.max - self.min

    @property
    def midpoint(self) -> float:
        """Midpoint between the minimum and maximum mapping value."""
        return (self.min + self.max) / 2

    def validate(self) -> None:
        """Checks the parameters for invalid settings."""
        super().validate()
        values = (self.min, self.max, self.percentile)
        if not all(isinstance(v, (int, float, type(None))) for v in values):
            raise TypeError(
                f"min, max, percentile must a float, int, or None. Found: "
                f"{type(self.min)=}, "
                f"{type(self.max)=}, "
                f"{type(self.percentile)=}, ",
            )
        if self.percentile is None:
            if self.max is None or self.min is None:
                raise ValueError("Min and max must be set if percentile is None.")
            if self.max <= self.min:
                raise ValueError("Max must be greater than min.")
        else:
            if self.min is not None or self.max is not None:
                raise ValueError("Min and max must be None if percentile is set.")


@dataclass(repr=False)
class WeightMappingParameters(MappingParameters):
    """Parameters for mapping outputs.

    Attributes:
        clipping: Whether to use clipping or not
        min: Minimum value before clipping
        max: Maximum value before clipping
        percentile: Percentile to clip at, if used
    """

    clipping: bool = True
    min: float = None
    max: float = None
    percentile: float = 1.0


# TODO: Improve naming?
@dataclass(repr=False)
class MatmulParameters(BasePairedParameters):
    """Parameters used to describe matrix operations.

    Attributes:
        _match: Whether or not to sync mvm or vmm settings.
            If true, mvm settings take precedence.
        mvm: Value mapping parameters for mvm operations
        vmm: Value mapping parameters for vmm operations

    Raises:
        ValueError: if match is True, but mvm and vmm are not equal
    """

    _match: bool = True
    mvm: MappingParameters = None
    vmm: MappingParameters = None


@dataclass(repr=False)
class CoreMappingParameters(BaseParameters):
    """Parameters for mapping values inside a core.

    Args:
        weights: Parameters for mapping values of weights
        inputs: Parameters for mapping values of inputs
    """

    weights: WeightMappingParameters = None
    inputs: MatmulParameters = None


def expand_key(raw_key: str) -> list[Any]:
    """Expands abbreviated key notation for core parameter subcores.

    A key of "0-3" will expand to [0, 1, 2, 3].
    A key of "2-1" will error.
    A key of "positive" will return ["positive"]
    """
    if not isinstance(raw_key, str):
        return [raw_key]
    if re.match("^\\s*\\d+\\s*$", raw_key):
        return [int(raw_key)]
    elif re.match("^\\s*\\d+\\s*-\\s*\\d+\\s*$", raw_key):
        start, _, finish = raw_key.partition("-")
        start = int(start)
        finish = int(finish)
        if start > finish:
            raise ValueError(
                "Improperly formatted shorthand key core subcore parameters. "
                "Subcore keys must either be an integer, or a string of the form "
                '"a-b" for integers a <= b',
            )
        return list(range(start, finish + 1))
    else:
        return [raw_key]
