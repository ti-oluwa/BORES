import numpy as np
import pytest

from bores.errors import ValidationError
from bores.schedule.summary import SummaryReport, as_dataframe

from .helpers import START_DATE, as_columns, is_categorical, make_summary_report


def test_wide_layout_has_a_row_for_each_time_and_a_column_for_each_vector(backend):
    frame = as_dataframe(make_summary_report(), backend=backend)
    columns = as_columns(frame)
    assert list(columns) == [
        "time",
        "date",
        "time_step",
        "FOPR",
        "WBHP:PROD1",
        "WBHP:INJ1",
        "FOPT",
    ]
    assert columns["time"] == [1.0, 2.0, 3.0]
    assert columns["FOPR"] == [99.0, 98.0, 97.0]
    assert columns["WBHP:INJ1"] == [4001.0, None, 4003.0]
    assert columns["FOPT"] == [None, None, 5.0]
    assert columns["time_step"] == [0, 1, 2]


def test_every_wide_column_matches_the_report_series(backend):
    report = make_summary_report()
    columns = as_columns(as_dataframe(report, backend=backend, include_dates=False))
    for key in report.keys():
        times, values = report.series(key)
        present = [
            (t, v) for t, v in zip(columns["time"], columns[key], strict=True) if v is not None
        ]
        assert [t for t, _ in present] == times.tolist()
        assert [v for _, v in present] == values.tolist()


def test_wide_layout_gaps_are_nulls_in_polars_and_nan_in_pandas(backend):
    frame = as_dataframe(make_summary_report(), backend=backend)
    if backend == "polars":
        assert frame["WBHP:INJ1"].null_count() == 1
    else:
        assert np.isnan(frame["WBHP:INJ1"].iloc[1])


def test_long_layout_has_a_row_for_each_record_in_recording_order(backend):
    report = make_summary_report()
    columns = as_columns(as_dataframe(report, backend=backend, layout="long"))
    assert list(columns) == ["key", "time", "value", "time_step", "date"]
    assert columns["key"] == [record.key for record in report.records]
    assert columns["value"] == [record.value for record in report.records]
    assert columns["time_step"][-1] is None
    assert columns["date"][-1] is None
    assert columns["date"][0].day == (START_DATE.day + 1)


def test_split_keys_separates_the_vector_from_its_qualifier(backend):
    columns = as_columns(
        as_dataframe(make_summary_report(), backend=backend, layout="long", split_keys=True)
    )
    pairs = set(zip(columns["vector"], columns["qualifier"], strict=True))
    assert pairs == {("FOPR", None), ("WBHP", "PROD1"), ("WBHP", "INJ1"), ("FOPT", None)}


def test_categorical_keys(backend):
    frame = as_dataframe(
        make_summary_report(), backend=backend, layout="long", split_keys=True, categorical=True
    )
    for name in ("key", "vector", "qualifier"):
        assert is_categorical(frame[name])


def test_keys_can_be_exact_or_patterns_and_keep_the_order_asked(backend):
    report = make_summary_report()
    columns = as_columns(
        as_dataframe(
            report,
            backend=backend,
            keys=["FOPT", "WBHP:*"],
            include_dates=False,
            include_time_steps=False,
        )
    )
    assert list(columns) == ["time", "FOPT", "WBHP:PROD1", "WBHP:INJ1"]
    single = as_columns(as_dataframe(report, backend=backend, keys="FOPR", include_dates=False))
    assert "FOPR" in single and "FOPT" not in single


def test_window_keeps_records_between_start_and_end_inclusive(backend):
    columns = as_columns(as_dataframe(make_summary_report(), backend=backend, start=2.0, end=3.0))
    assert columns["time"] == [2.0, 3.0]
    assert columns["FOPR"] == [98.0, 97.0]
    long = as_columns(
        as_dataframe(
            make_summary_report(), backend=backend, layout="long", keys="WBHP:*", start=3.0
        )
    )
    assert long["key"] == ["WBHP:PROD1", "WBHP:INJ1"]


def test_date_and_step_columns_follow_the_records_unless_told_otherwise(backend):
    report = SummaryReport()
    report.record(key="A", time=1.0, value=1.0)
    assert list(as_columns(as_dataframe(report, backend=backend))) == ["time", "A"]
    forced = as_columns(
        as_dataframe(report, backend=backend, include_dates=True, include_time_steps=True)
    )
    assert list(forced) == ["time", "date", "time_step", "A"]
    assert forced["date"] == [None] and forced["time_step"] == [None]


def test_empty_report_gives_empty_frames(backend):
    assert as_dataframe(SummaryReport(), backend=backend).shape == (0, 1)
    assert as_dataframe(SummaryReport(), backend=backend, layout="long").shape == (0, 3)


def test_window_with_no_records_is_empty_but_keeps_the_vectors(backend):
    frame = as_dataframe(make_summary_report(), backend=backend, start=100.0)
    assert frame.shape[0] == 0 and "FOPR" in frame.columns


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"keys": "NOPE"}, "matches"),
        ({"keys": "Q*"}, "matches"),
        ({"keys": ["FOPR", "NOPE"]}, "matches"),
        ({"layout": "tall"}, "layout"),
        ({"split_keys": True}, "long"),
        ({"categorical": True}, "long"),
        ({"start": 5.0, "end": 1.0}, "start"),
        ({"backend": "arrow"}, "backend"),
    ],
)
def test_invalid_requests_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        as_dataframe(make_summary_report(), **kwargs)


def test_the_method_gives_the_same_frame_as_the_function(backend):
    report = make_summary_report()
    assert as_columns(report.as_dataframe(backend=backend, layout="long")) == as_columns(
        as_dataframe(report, backend=backend, layout="long")
    )
