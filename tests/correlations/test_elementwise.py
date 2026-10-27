import numba
import numpy as np
import pytest

from bores.correlations.elementwise import Elementwise, elementwise
from bores.errors import ValidationError


@elementwise
def double(x, offset=1.0):
    return 2.0 * x + offset


@elementwise
def chained(x, y):
    return double(x, 0.5) * y


@elementwise(positive=("x",), non_negative=("y",), bounds={"z": (0.0, 1.0)})
def checked(x, y, z):
    return x + y + z


def check_two_arguments(values):
    if np.max(values["x"] + values["y"]) > 10:
        raise ValidationError("sum too large")


@elementwise(validators=(check_two_arguments,))
def validated(x, y):
    return x + y


def test_decorator_returns_elementwise():
    assert isinstance(double, Elementwise)


def test_scalar_in_scalar_out():
    result = double(3.0, 2.0)
    assert np.ndim(result) == 0
    assert result == pytest.approx(8.0)


def test_array_in_array_out_and_broadcast():
    x = np.arange(6.0).reshape(2, 3)
    result = double(x, 1.0)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, 2 * x + 1)
    broadcast = chained(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(broadcast, (2 * x + 0.5) * np.array([1.0, 2.0, 3.0]))


def test_keyword_positional_and_mixed_calls_agree():
    assert double(2.0, offset=3.0) == double(2.0, 3.0) == double(x=2.0, offset=3.0)


def test_defaults_are_filled():
    assert double(2.0) == pytest.approx(5.0)
    assert double(x=2.0) == pytest.approx(5.0)


def test_bind_errors():
    with pytest.raises(TypeError, match="missing required argument 'y'"):
        chained(1.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'nope'"):
        double(1.0, nope=2.0)
    with pytest.raises(TypeError, match="multiple values for argument 'x'"):
        double(1.0, x=2.0)
    with pytest.raises(TypeError, match=r"takes 2 arguments but 3 were given"):
        double(1.0, 2.0, 3.0)


def test_float32_arrays_stay_float32():
    result = double(np.ones(4, dtype=np.float32), np.float32(1.0))
    assert result.dtype == np.float32
    assert double(np.ones(4), 1.0).dtype == np.float64


def test_callable_from_njit_code():
    @numba.njit
    def use(x, y):
        return chained(x, y) + double(x, 1.0)

    assert use(2.0, 3.0) == pytest.approx(chained(2.0, 3.0) + double(2.0, 1.0))


def test_positive_non_negative_and_bounds_checks():
    assert checked(1.0, 0.0, 0.5) == pytest.approx(1.5)
    with pytest.raises(ValidationError, match="`x` must be greater than zero"):
        checked(0.0, 1.0, 0.5)
    with pytest.raises(ValidationError, match="`y` must be non-negative"):
        checked(1.0, -1.0, 0.5)
    with pytest.raises(ValidationError, match=r"`z` must be between 0\.0 and 1\.0"):
        checked(1.0, 1.0, 1.5)


def test_checks_apply_to_every_array_element():
    with pytest.raises(ValidationError):
        checked(np.array([1.0, 2.0, -3.0]), 1.0, 0.5)


def test_multi_argument_validator():
    assert validated(1.0, 2.0) == pytest.approx(3.0)
    with pytest.raises(ValidationError, match="sum too large"):
        validated(8.0, 8.0)


def test_unknown_checked_argument_is_rejected_at_definition():
    with pytest.raises(ValueError, match="not parameters"):

        @elementwise(positive=("missing",))
        def broken(x):
            return x


class TestResultDtype:
    def test_resolve_result_dtype_follows_numpy_promotion(self):
        from bores.correlations.elementwise import resolve_result_dtype

        f32 = np.ones(2, dtype=np.float32)
        assert resolve_result_dtype(f32, 2.0) == np.float32
        assert resolve_result_dtype(f32, np.float64(2.0)) == np.float64
        assert resolve_result_dtype(f32, np.ones(2)) == np.float64
        assert resolve_result_dtype(np.ones(2, dtype=np.float16), f32) == np.float32
        assert resolve_result_dtype(np.ones(2, dtype=np.int32), 1.0) == np.float64
        assert resolve_result_dtype([1.0, 2.0], f32) == np.float64
        assert resolve_result_dtype(1.0, 2.0) == np.float64

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
    def test_floating_inputs_keep_their_dtype(self, dtype):
        result = double(np.ones(3, dtype=dtype), 1.0)
        assert result.dtype == dtype
        np.testing.assert_allclose(result.astype(np.float64), 3.0, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.bool_])
    def test_integer_and_boolean_inputs_give_float64(self, dtype):
        assert double(np.ones(3, dtype=dtype), 1.0).dtype == np.float64

    def test_numpy_float64_scalar_promotes_a_float32_array(self):
        assert double(np.ones(3, dtype=np.float32), np.float64(1.0)).dtype == np.float64

    def test_scalar_numpy_float32_gives_float32(self):
        assert double(np.float32(2.0), np.float32(1.0)).dtype == np.float32


@elementwise(target="parallel")
def parallel_scaled(x, y):
    return 3.0 * x + y


@elementwise(cache=False, numba_options={"fastmath": True})
def uncached_fast(x):
    return x * 2.0


class TestCompilationOptions:
    def test_options_are_exposed(self):
        assert parallel_scaled.target == "parallel"
        assert parallel_scaled.cache is True
        assert uncached_fast.cache is False
        assert uncached_fast.numba_options == {"fastmath": True}
        assert double.target == "cpu"

    def test_parallel_target_matches_the_cpu_target(self):
        x = np.linspace(0.0, 10.0, 1001)
        np.testing.assert_allclose(parallel_scaled(x, 2.0), 3.0 * x + 2.0)
        np.testing.assert_allclose(parallel_scaled(x, x), 4.0 * x)

    def test_parallel_target_still_accepts_scalars_and_keywords(self):
        assert parallel_scaled(2.0, 1.0) == pytest.approx(7.0)
        assert parallel_scaled(y=1.0, x=np.array([1.0, 2.0]))[1] == pytest.approx(7.0)

    def test_parallel_target_keeps_dtypes(self):
        x = np.ones(5, dtype=np.float32)
        assert parallel_scaled(x, np.float32(1.0)).dtype == np.float32
        assert parallel_scaled(np.ones(5, dtype=np.float16), 1.0).dtype == np.float16
        assert parallel_scaled(np.ones(5, dtype=np.int64), 1.0).dtype == np.float64

    def test_numba_options_are_applied(self):
        np.testing.assert_allclose(uncached_fast(np.arange(4.0)), 2.0 * np.arange(4.0))

    def test_unknown_target_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown target"):

            @elementwise(target="gpu")  # type: ignore[call-overload]
            def broken(x):
                return x
