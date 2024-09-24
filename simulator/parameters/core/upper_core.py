#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from typing import Any, TypeVar
from dataclasses import dataclass, field

from simulator.parameters.core.core import CoreParameters, expand_key
from simulator.parameters.core.lower_core import (
    UnsignedCoreParameters,
)

T = TypeVar("T")


@dataclass(repr=False)
class OffsetCoreParameters(CoreParameters):
    """Parameters for describing offset core behavior.

    Args:
        core_type: Type of core associated with the core parameter
        style: Style of the offset core
    """

    core_type: str = "OffsetCore"
    subcores: dict[Any, CoreParameters] = field(
        default_factory=lambda: {0: UnsignedCoreParameters()},
    )


@dataclass(repr=False)
class BitSlicedCoreParameters(CoreParameters):
    """Parameters for describing bit sliced core behavior.

    Args:
        style: Style of the bit sliced core
        num_slices: Number of slices in the core
        Nbits_reduction: Number of bits by which to reduce the ADC range of the
            top bit slice
    """

    core_type: str = "BitslicedCore"
    slice_size: int | tuple[int] = None

    @property
    def style(self):
        """Returns the style of the bitsliced subcores."""
        if self.subcores is None:
            raise ValueError("BitSlicedCoreParameters requires subcores to be defined.")
        style_ = {sc.core_type for sc in self.subcores}
        if len(style_) != 1:
            raise ValueError(
                "All subcores of BitSlicedCoreParameters must be of same core type",
            )
        return style_.pop()

    def __post_init__(self):
        """Runs after dataclass initialization."""
        super().__post_init__()
        if self.subcores is not None and not isinstance(self.slice_size, tuple):
            keys = []
            for raw_key in self.subcores.keys():
                keys.extend(expand_key(raw_key=raw_key))
            self.slice_size = tuple([self.slice_size] * len(keys))
        self.validate()

    def validate(self):
        """Checks the parameters for invalid settings."""
        super().validate()
        if self.subcores is None:
            return
        raw_keys = list(self.subcores.keys())
        keys = []
        for raw_key in raw_keys:
            keys.extend(expand_key(raw_key=raw_key))

        if len(keys) != len(set(keys)):
            raise KeyError(
                f"Expanded keys have overlapped keys:\n"
                f"Raw keys: {raw_keys}\n"
                f"Expanded keys: {keys}",
            )

        key_min, key_max, key_len = min(keys), max(keys), len(keys)

        inferred_slice_size = False
        slice_size = self.slice_size
        if isinstance(slice_size, int):
            slice_size = [slice_size]
            inferred_slice_size = True

        if key_min != 0 or (key_max - key_min + 1) != key_len:
            raise ValueError(
                "BitSlicedCoreParameters subcore keys must include all integers 0-n. ",
                f"Got keys: {keys = }",
            )
        if not all(isinstance(k, int) for k in keys):
            raise ValueError(
                "BitSlicedCoreParameters subcore keys must be integers or be able ",
                'to be expanded to integers (e.g. "0-3"). ',
                f"Got keys: {list(keys)}.",
            )
        if not all(isinstance(s, int) for s in slice_size):
            raise ValueError(
                "Slice size must be a positive integer or list of positive integers. ",
                f"Got {self.slice_size = }",
            )
        elif any(s <= 0 for s in slice_size):
            raise ValueError(
                "Slice size must be a positive integer or list of positive integers. ",
                f"Got {self.slice_size = }",
            )
        elif not inferred_slice_size and len(self.slice_size) != len(keys):
            raise ValueError(
                "Slice size length must match subcore length. ",
                f"Got {len(self.slice_size) = }, ",
                f"{len(keys) = }. ",
            )
