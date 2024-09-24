#
# Copyright 2017-2023 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from __future__ import annotations

from typing import Any, get_type_hints, TypeVar

from simulator.backend.glob import globfilter
from simulator.backend.registry import RegistryManager
from .parameters import BaseParameters

T = TypeVar("T")


def flatten_param(param: BaseParameters | dict, **kwargs) -> dict[str, Any]:
    """Converts a parameter into a flat dictionary.

    Args:
        param: Parameter to flatten.
        **kwargs: Keyword arguments to pass to param.as_dict

    Returns:
        dict[str, Any]: Flattened dictionary representation of the parameter.
    """
    if isinstance(param, BaseParameters):
        return flatten_param(param.as_dict(**kwargs))
    param: dict
    flat = {}
    for key, value in param.items():
        if isinstance(value, (dict, BaseParameters)):
            subflat = flatten_param(value)
            for k, v in subflat.items():
                flat[f"{key}.{k}"] = v
        else:
            flat[str(key)] = value
    return flat


def nest_dict(param: dict[str, Any]) -> dict[str, Any]:
    """Nests a flattened dictionary. Nesting occurs on dotted keys.
    (e.g. foo.bar.baz).

    Args:
        param: Flattened dictionary to nest.

    Returns:
        dict[str, Any]: A nested version of a flattened parameter.
    """
    # Base case
    nested = {k: v for k, v in param.items() if "." not in k}

    # Recursive case
    needs_nesting = {k: v for k, v in param.items() if "." in k}
    prefixes = {str(k).partition(".")[0] for k in needs_nesting.keys()}
    for prefix in prefixes:
        prefix_dict = {}
        for k, v in param.items():
            if str(k).split(".")[0] != prefix:
                continue
            postfix = str(k).partition(".")[-1]
            prefix_dict[postfix] = v
        nested[prefix] = nest_dict(prefix_dict)
    return nested


def resolve_registered_and_builtin_types(t: str | type) -> type:
    """Resolves the type of an input.

    Args:
        t: The type to resolve. If t is already a type, it is returned as is.
            If t is a string, an attempt will be made to fetch a type by the
            same name will be fetched from the RegistryManager, or by looking
            at built in types.

    Returns:
        type: The resolved type from the input.
    """
    if isinstance(t, type):
        return t
    if not isinstance(t, str):
        raise TypeError("resolve_type expects either a string or a type.")

    registry_manager = RegistryManager()
    dummy_type = type("_", (), {"__annotations__": {"type": t}})
    localns = {}
    for k, v in registry_manager.items():
        localns.update(**{**v, k.__name__: k})
    try:
        resolved_type = get_type_hints(dummy_type, localns=localns)["type"]
    except NameError as e:
        raise NameError(
            f"Could not resolve the type '{e.name}', it is not registered or a builtin",
        ) from e
    return resolved_type


def get_matching_keys(param: BaseParameters, key: str) -> set[str]:
    """Returns a set of all keys that match on the parameter.

    Args:
        param: Parameter to get matching keys on
        key: Key to match on for setting values.

    Returns:
        set[str]: Set of keys that match the search key
    """
    search_keys = key.split(",")
    keys = set()
    for child_node in param.as_dict(flat=True).keys():
        child_key = child_node
        while child_key != "":
            keys.add(child_key)
            child_key, _, _ = child_key.rpartition(".")

    matching_keys = set()
    for search_key in search_keys:
        search_result = globfilter(keys, search_key)
        matching_keys.update(search_result)
    return matching_keys


def convert_type(
    value_type: T,
    value: Any,
    key_name: str,
    parent: BaseParameters,
) -> T:
    """Attempts to convert a value to a specified type.

    Args:
        value_type: Type to convert value to
        value: Value to convert
        key_name: Name of key that is being converted
        parent: Parent parameter where the value is being converted.

    Raises:
        ValueError: Raised if a value couldn't be converted.

    Returns:
        T: Returns an object of specified type.
    """
    registry_manager = RegistryManager()
    value_type = registry_manager.get_from_root(value_type)
    try:
        value = registry_manager.convert(value_type, value)
        if isinstance(value, BaseParameters):
            value._parent = parent
        return value
    except (ValueError, KeyError) as e:
        if not isinstance(parent, BaseParameters):
            key = key_name
        else:
            # Helps the user find where they incorrectly configured parameters
            if parent.root is parent:
                key = f"{key_name}"
            else:
                key = f"{parent.get_path_from_root()}.{key_name}"
        conversion_error_msg = "\n".join(e.args)
        error_msg = (
            f"Could not convert key '{key}' to type '{value_type.__name__}' "
            f"using '{key} = {value}'.\n"
            f"Reason:\n\n"
            f"{conversion_error_msg}"
        )
        raise ValueError(error_msg) from e
