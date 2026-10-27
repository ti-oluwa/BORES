import inspect
import typing

import numpy as np
from numba.np.ufunc.dufunc import DUFunc  # type: ignore[import-untyped]

from bores.errors import ValidationError
from bores.types import NDimension, Number, NumberOrArray

__all__ = ["Elementwise", "elementwise"]


class Elementwise(DUFunc):
    """
    A correlation written once as a scalar function and usable on scalars and arrays alike.

    Calling it with plain numbers returns a number. Calling it with arrays of any
    shape (broadcast against each other and against scalars) returns an array.
    Arguments can be passed by keyword or by position. The same object can be
    called from inside other `numba.njit` functions and `elementwise` correlations
    with positional arguments.

    Float32 array inputs produce a float32 result, otherwise the result follows
    NumPy's usual promotion rules.

    Argument checks declared on the decorator run once per call, on the whole
    input, before any computation.
    """

    def __init__(
        self,
        py_func: typing.Callable[..., typing.Any],
        *,
        positive: typing.Iterable[str] = (),
        non_negative: typing.Iterable[str] = (),
        bounds: typing.Mapping[str, tuple[Number, Number]] | None = None,
        validators: typing.Iterable[typing.Callable[[dict[str, typing.Any]], None]] = (),
    ) -> None:
        """
        :param py_func: Scalar implementation. Must be compilable by `numba.njit`.
        :param positive: Names of arguments that must be strictly greater than zero.
        :param non_negative: Names of arguments that must be greater than or equal to zero.
        :param bounds: Names of arguments mapped to their inclusive `(lower, upper)` range.
        :param validators: Functions that receive every argument by name and raise
            `ValidationError` for invalid combinations of arguments.
        """
        super().__init__(py_func, identity=None, cache=True, targetoptions={})
        self.py_func = py_func
        parameters = inspect.signature(py_func).parameters
        self.argument_names = tuple(parameters)
        self.defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        self.positive = tuple(positive)
        self.non_negative = tuple(non_negative)
        self.bounds = dict(bounds or {})
        self.validators = tuple(validators)
        unknown = {
            *self.positive,
            *self.non_negative,
            *self.bounds,
        } - set(self.argument_names)
        if unknown:
            raise ValueError(
                f"Checked arguments {sorted(unknown)} are not parameters of {py_func.__name__!r}."
            )

    def bind(self, args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]) -> tuple:
        """
        Orders positional and keyword arguments into the positional order of the scalar
        function, filling omitted arguments from the defaults of the scalar function.

        :param args: Positional arguments of the call.
        :param kwargs: Keyword arguments of the call.
        :return: All arguments in positional order.
        :raises TypeError: If an argument is missing, repeated or unknown.
        """
        names = self.argument_names
        if len(args) > len(names):
            raise TypeError(
                f"{self.__name__}() takes {len(names)} arguments but {len(args)} were given."
            )
        if len(args) == len(names) and not kwargs:
            return args

        repeated = set(names[: len(args)]) & set(kwargs)
        if repeated:
            raise TypeError(
                f"{self.__name__}() got multiple values for argument {sorted(repeated)[0]!r}."
            )

        unknown = set(kwargs) - set(names)
        if unknown:
            raise TypeError(
                f"{self.__name__}() got an unexpected keyword argument {sorted(unknown)[0]!r}."
            )

        filled = list(args)
        for name in names[len(args) :]:
            if name in kwargs:
                filled.append(kwargs[name])
            elif name in self.defaults:
                filled.append(self.defaults[name])
            else:
                raise TypeError(f"{self.__name__}() missing required argument {name!r}.")
        return tuple(filled)

    def check(self, values: tuple[typing.Any, ...]) -> None:
        """
        Validates the declared argument ranges.

        :param values: All arguments in positional order.
        :raises ValidationError: If any element of a checked argument is outside its range.
        """
        by_name = dict(zip(self.argument_names, values, strict=False))
        for name in self.positive:
            value = by_name.get(name)
            if value is not None and np.min(value) <= 0:
                raise ValidationError(f"`{name}` must be greater than zero.")

        for name in self.non_negative:
            value = by_name.get(name)
            if value is not None and np.min(value) < 0:
                raise ValidationError(f"`{name}` must be non-negative.")

        for name, (lower, upper) in self.bounds.items():
            value = by_name.get(name)
            if value is not None and (np.min(value) < lower or np.max(value) > upper):
                raise ValidationError(f"`{name}` must be between {lower} and {upper}.")

        for validator in self.validators:
            validator(by_name)

    def __call__(
        self, *args: NumberOrArray[NDimension], **kwargs: NumberOrArray[NDimension]
    ) -> NumberOrArray[NDimension]:  # type: ignore[override]
        values = self.bind(args=args, kwargs=kwargs)
        if self.positive or self.non_negative or self.bounds or self.validators:
            self.check(values)

        result = super().__call__(*values)
        if isinstance(result, np.ndarray) and result.dtype == np.float64:
            array_dtypes = [v.dtype for v in values if isinstance(v, np.ndarray)]
            if array_dtypes and np.result_type(*array_dtypes) == np.float32:
                return result.astype(np.float32, copy=False)
        return result


def elementwise(
    func: typing.Callable[..., typing.Any] | None = None,
    *,
    positive: typing.Iterable[str] = (),
    non_negative: typing.Iterable[str] = (),
    bounds: typing.Mapping[str, tuple[Number, Number]] | None = None,
    validators: typing.Iterable[typing.Callable[[dict[str, typing.Any]], None]] = (),
) -> typing.Any:
    """
    Turns a scalar function into an `Elementwise` correlation.

    Usable bare (`@elementwise`) or with argument checks
    (`@elementwise(positive=("pressure",), bounds={"porosity": (0.0, 1.0)})`).

    :param func: Scalar implementation, when used without arguments.
    :param positive: Names of arguments that must be strictly greater than zero.
    :param non_negative: Names of arguments that must be greater than or equal to zero.
    :param bounds: Names of arguments mapped to their inclusive `(lower, upper)` range.
    :param validators: Functions that receive every argument by name and raise
        `ValidationError` for invalid combinations of arguments.
    :return: The `Elementwise` correlation, or a decorator that builds one.
    """

    def decorate(function: typing.Callable[..., typing.Any]) -> Elementwise:
        return Elementwise(
            function,
            positive=positive,
            non_negative=non_negative,
            bounds=bounds,
            validators=validators,
        )

    if func is None:
        return decorate
    return decorate(func)
