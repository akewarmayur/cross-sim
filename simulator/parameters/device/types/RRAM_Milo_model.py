#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

import logging
from dataclasses import dataclass

from simulator.parameters.device.device import (
    DeviceModelParameters,
)

log = logging.getLogger(__name__)


@dataclass(repr=False)
class RRAMMiloParameters(DeviceModelParameters):
    """Parameters for the RRAM Milo device model.

    Args:
        model: The class of device this parameter is for.
    """

    model: str = "RRAMMilo"
