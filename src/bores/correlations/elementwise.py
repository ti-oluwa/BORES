import inspect
import typing
from collections.abc import Callable, Iterable, Mapping

import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from numba.np.ufunc.dufunc import DUFunc  # type: ignore[import-untyped]

from bores.errors import ValidationError
from bores.types import Number, NumberOrArray

__all__ = ["Elementwise", "elementwise", "resolve_result_dtype"]

P = typing.ParamSpec("P")
R = typing.TypeVar("R")

Target = typing.Literal["cpu", "parallel"]
"""Where the compiled loop runs: "cpu" on one thread, "parallel" across threads."""

NumbaOption = bool | int | float | str
"""Value of an option passed through to numba, such as `fastmath=True`."""

Validator = Callable[[Mapping[str, NumberOrArray]], None]
"""Check on the arguments of a call, by name, that raises `ValidationError` when they are invalid."""

COMPILED_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
"""Floating point dtypes the compiled correlations compute in."""

PARALLEL_SIGNATURE_TYPES = ("float64", "float32")
"""Argument and result types a parallel correlation is compiled for."""


def resolve_result_dtype(*values: NumberOrArray | list[float] | tuple[float, ...]) -> np.dtype:
    """
    Resolves the floating point dtype of a result from its inputs with NumPy's promotion rules.

    Arrays and NumPy scalars set the dtype. Python numbers adapt to it, so a float32
    array combined with a Python float gives float32, while a float32 array combined
    with a float64 array or NumPy float64 scalar gives float64. Integer and boolean
    inputs give float64.

    :param values: Numbers, arrays, or sequences of numbers.
    :return: The floating point dtype of the result.
    """
    operands = [np.asarray(v) if isinstance(v, (list, tuple)) else v for v in values]
    dtype = np.result_type(*operands)
    return dtype if dtype.kind == "f" else np.dtype(np.float64)


def prepare_operand(value: NumberOrArray) -> NumberOrArray:
    """
    Converts floating point operands the compiled code cannot take (float16 and
    extended precision floats) to float64, and leaves everything else unchanged.

    :param value: A number or array.
    :return: The operand in a dtype the compiled code accepts.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f" and value.dtype not in COMPILED_FLOAT_DTYPES:
            return value.astype(np.float64)
    elif isinstance(value, np.floating) and value.dtype not in COMPILED_FLOAT_DTYPES:
        return np.float64(value)
    return value


class Elementwise(DUFunc, typing.Generic[P, R]):  # type: ignore[misc]
    """
    A correlation written once as a scalar function and usable on scalars and arrays alike.

    Calling it with plain numbers returns a number. Calling it with arrays of any
    shape (broadcast against each other and against scalars) returns an array.
    Arguments can be passed by keyword or by position. The same object can be
    called from inside other `numba.njit` functions and `elementwise` correlations
    with positional arguments.

    The object is generic over the parameters `P` and the return type `R` of the scalar
    function it wraps, so a type checker sees the original argument names. Annotate
    the scalar function with `NumberOrArray`, since that is what callers may pass and
    receive.

    The result has the floating point dtype NumPy's promotion rules give for the
    inputs: float32 arrays give float32, float64 gives float64, and integers give
    float64. Float16 and extended precision inputs are computed in float64 and the
    result is returned in the input dtype.

    Argument checks declared on the decorator run once per call, on the whole
    input, before any computation.
    """

    def __init__(
        self,
        py_func: Callable[P, R],
        *,
        positive: Iterable[str] = (),
        non_negative: Iterable[str] = (),
        bounds: Mapping[str, tuple[Number, Number]] | None = None,
        validators: Iterable[Validator] = (),
        cache: bool = True,
        target: Target = "cpu",
        numba_options: Mapping[str, NumbaOption] | None = None,
    ) -> None:
        """
        :param py_func: Scalar implementation. Must be compilable by `numba.njit`.
        :param positive: Names of arguments that must be strictly greater than zero.
        :param non_negative: Names of arguments that must be greater than or equal to zero.
        :param bounds: Names of arguments mapped to their inclusive `(lower, upper)` range.
        :param validators: Functions that receive every argument by name and raise
            `ValidationError` for invalid combinations of arguments.
        :param cache: Whether to cache the compiled code on disk between sessions.
        :param target: "cpu" runs the loop on one thread. "parallel" runs it across
            threads when called with arrays, which pays off for large arrays. Calls
            with only numbers always run on one thread. The parallel loop is compiled
            for float64 and float32 on first use.
        :param numba_options: Options passed through to numba when compiling, such as
            `{"fastmath": True}`.
        :raises ValueError: If `target` is not recognized or a checked argument is
            not a parameter of `py_func`.
        """
        if target not in ("cpu", "parallel"):
            raise ValueError(f"Unknown target {target!r}. Use 'cpu' or 'parallel'.")
        self.numba_options = dict(numba_options or {})
        super().__init__(
            py_func, identity=None, cache=cache, targetoptions=dict(self.numba_options)
        )
        self.py_func = py_func
        self.cache = cache
        self.target = target
        parameters = inspect.signature(py_func).parameters
        self.argument_names = tuple(parameters)
        self.defaults: dict[str, Number] = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        self.positive = tuple(positive)
        self.non_negative = tuple(non_negative)
        self.bounds = dict(bounds or {})
        self.validators = tuple(validators)
        self.parallel_loop: np.ufunc | None = None
        unknown = {*self.positive, *self.non_negative, *self.bounds} - set(self.argument_names)
        if unknown:
            raise ValueError(
                f"Checked arguments {sorted(unknown)} are not parameters of {py_func.__name__!r}."
            )

    def get_parallel_loop(self) -> np.ufunc:
        """
        Returns the multi-threaded loop, compiling it on first use.

        :return: A universal function over float64 or float32 arrays.
        """
        if self.parallel_loop is None:
            arity = len(self.argument_names)
            signatures = [
                f"{dtype}({', '.join([dtype] * arity)})" for dtype in PARALLEL_SIGNATURE_TYPES
            ]
            self.parallel_loop = numba.vectorize(
                signatures, target="parallel", cache=self.cache, **self.numba_options
            )(self.py_func)
        return typing.cast(np.ufunc, self.parallel_loop)

    def bind(
        self,
        args: tuple[NumberOrArray, ...],
        kwargs: Mapping[str, NumberOrArray],
    ) -> tuple[NumberOrArray, ...]:
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

    def check(self, values: tuple[NumberOrArray, ...]) -> None:
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

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        values = self.bind(
            args=typing.cast(tuple[NumberOrArray, ...], args),
            kwargs=typing.cast(Mapping[str, NumberOrArray], kwargs),
        )
        if self.positive or self.non_negative or self.bounds or self.validators:
            self.check(values)

        dtype = resolve_result_dtype(*values)
        operands = tuple(prepare_operand(v) for v in values)
        if self.target == "parallel" and any(isinstance(o, np.ndarray) for o in operands):
            compute_dtype = np.float32 if dtype == np.float32 else np.float64
            result = self.get_parallel_loop()(
                *(np.asarray(o, dtype=compute_dtype) for o in operands)
            )
        else:
            result = super().__call__(*operands)

        result = typing.cast(npt.NDArray | np.generic, result)
        if result.dtype != dtype:
            result = result.astype(dtype, copy=False)
        return typing.cast(R, result)


@typing.overload
def elementwise(func: Callable[P, R], /) -> Elementwise[P, R]: ...
@typing.overload
def elementwise(
    *,
    positive: Iterable[str] = ...,
    non_negative: Iterable[str] = ...,
    bounds: Mapping[str, tuple[Number, Number]] | None = ...,
    validators: Iterable[Validator] = ...,
    cache: bool = ...,
    target: Target = ...,
    numba_options: Mapping[str, NumbaOption] | None = ...,
) -> Callable[[Callable[P, R]], Elementwise[P, R]]: ...


def elementwise(
    func: Callable[P, R] | None = None,
    /,
    *,
    positive: Iterable[str] = (),
    non_negative: Iterable[str] = (),
    bounds: Mapping[str, tuple[Number, Number]] | None = None,
    validators: Iterable[Validator] = (),
    cache: bool = True,
    target: Target = "cpu",
    numba_options: Mapping[str, NumbaOption] | None = None,
) -> Elementwise[P, R] | Callable[[Callable[P, R]], Elementwise[P, R]]:
    """
    Turns a scalar function into an `Elementwise` correlation.

    Usable bare (`@elementwise`) or with options
    (`@elementwise(positive=("pressure",), bounds={"porosity": (0.0, 1.0)}, target="parallel")`).

    :param func: Scalar implementation, when used without options.
    :param positive: Names of arguments that must be strictly greater than zero.
    :param non_negative: Names of arguments that must be greater than or equal to zero.
    :param bounds: Names of arguments mapped to their inclusive `(lower, upper)` range.
    :param validators: Functions that receive every argument by name and raise
        `ValidationError` for invalid combinations of arguments.
    :param cache: Whether to cache the compiled code on disk between sessions.
    :param target: "cpu" runs the loop on one thread. "parallel" runs it across
        threads when called with arrays.
    :param numba_options: Options passed through to numba when compiling, such as
        `{"fastmath": True}`.
    :return: The `Elementwise` correlation, or a decorator that builds one.
    """

    def decorate(function: Callable[P, R]) -> Elementwise[P, R]:
        return Elementwise(
            function,
            positive=positive,
            non_negative=non_negative,
            bounds=bounds,
            validators=validators,
            cache=cache,
            target=target,
            numba_options=numba_options,
        )

    if func is None:
        return decorate
    return decorate(func)
