import argparse
import inspect
from typing import Callable, Tuple, List, Any, Dict, get_type_hints

from typing import Union, Optional, get_origin, get_args, Type

def unwrap_type(tp: Type) -> Type:
    """
    Extracts the underlying type from Optional[T] or Union[T, NoneType].
    Returns the original type if not Optional.
    """
    origin = get_origin(tp)
    args = get_args(tp)

    # Check for Optional[T] (which is Union[T, None])
    if origin is Union and len(args) == 2 and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        return non_none[0] if non_none else str  # default to str if empty
    return tp  # Return as-is if not Optional/Union


# def generate_argparser_from_func(
#         func: Callable,
#         parser: Optional[argparse.ArgumentParser] = None
# ) -> Tuple[argparse.ArgumentParser, List[str]]:
#     """
#     Adds CLI arguments to an argparse.ArgumentParser based on the signature of the given function.
#
#     Args:
#         func: The function whose parameters should be converted into CLI arguments.
#         parser: An optional existing ArgumentParser to add arguments to.
#
#     Returns:
#         A tuple of:
#         - The updated ArgumentParser
#         - A list of argument names (strings) that were added automatically based on the function's signature
#     """
#     if parser is None:
#         parser = argparse.ArgumentParser(description=f"Auto-generated CLI for {func.__name__}")
#
#     existing_args = {a.dest for a in parser._actions}
#     sig = inspect.signature(func)
#     type_hints = get_type_hints(func)
#     added_args: List[str] = []
#
#     for name, param in sig.parameters.items():
#         if name in existing_args:
#             continue  # Skip args already added
#
#         param_type = type_hints.get(name, str)
#         default = param.default
#
#         arg_name = f"--{name.replace('_', '-')}"
#         kwargs = {
#             "help": f"(inferred type: {getattr(param_type, '__name__', str(param_type))})",
#             "dest": name,
#         }
#
#         if param_type == bool:
#             if default is False:
#                 parser.add_argument(arg_name, action="store_true", **kwargs)
#             elif default is True:
#                 parser.add_argument(arg_name, action="store_false", **kwargs)
#             else:
#                 parser.add_argument(arg_name, type=bool, required=False, **kwargs)
#         else:
#             kwargs["type"] = unwrap_type(param_type)
#             parser.add_argument(arg_name, required=False, **kwargs)
#             # if default is not inspect.Parameter.empty:
#             #     kwargs["default"] = default
#             #     parser.add_argument(arg_name, **kwargs)
#             # else:
#             #     parser.add_argument(arg_name, required=False, **kwargs)
#
#         added_args.append(name)
#
#     return parser, added_args


def get_function_param_names(func: Callable) -> List[str]:
    """
    Returns the list of parameter names for a given function.

    Args:
        func: The function whose parameters to extract.

    Returns:
        A list of parameter names (as strings), in the order they appear in the signature.
    """
    return list(inspect.signature(func).parameters)


def extract_kwargs_from_args(
    all_args: dict,
    include_keys: List[str]
) -> Dict[str, Any]:
    """
    Extracts a dictionary of keyword arguments from an argparse.Namespace,
    limited to the keys in `include_keys`, excluding keys with value None.

    Args:
        args: Parsed arguments from argparse.
        include_keys: List of argument names to extract from args.

    Returns:
        Dictionary of {name: value} for each matching key in args,
        excluding entries with value None.
    """
    return {
        k: v for k, v in all_args.items()
        if k in include_keys and v is not None
    }


def parse_unknown_args(unknown_args) -> dict:
    """
    Parse a list like ["--foo", "bar", "--baz", "2"] into a dict {foo: bar, baz: 2}
    Converts numeric values to int or float where possible.
    """
    parsed = {}
    key = None
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            parsed[key] = True  # default to True for flags
        else:
            if key is None:
                continue  # ignore values without a preceding flag
            # Try to convert value to int or float
            try:
                val = int(arg)
            except ValueError:
                try:
                    val = float(arg)
                except ValueError:
                    val = arg
            parsed[key] = val
            key = None
    return parsed
