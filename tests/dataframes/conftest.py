import pytest

BACKENDS = ["pandas", "polars"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Each data frame library in turn, skipping a library that is not installed."""
    pytest.importorskip(request.param)
    return request.param
