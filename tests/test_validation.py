import pyarrow as pa
import pyarrow.parquet as pq

from lobpy.cli import main as cli_main
from lobpy.validation import validate_parquet


def _write_events(tmp_path, rows, *, fields=None):
    fields = fields or [
        pa.field("timestamp", pa.int64()),
        pa.field("event_type", pa.string()),
        pa.field("side", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("quantity", pa.float64()),
    ]
    data = {
        "timestamp": [r[0] for r in rows],
        "event_type": [r[1] for r in rows],
        "side": [r[2] for r in rows],
        "price": [r[3] for r in rows],
        "quantity": [r[4] for r in rows],
    }
    path = tmp_path / "events.parquet"
    pq.write_table(pa.Table.from_pydict(data, schema=pa.schema(fields)), path)
    return path


def _codes(result):
    return {issue.code for issue in result.errors}


def test_validate_parquet_valid_file(tmp_path):
    path = _write_events(
        tmp_path,
        [
            (1000, "book_level", "bid", 100.0, 5.0),
            (1000, "book_level", "ask", 101.0, 3.0),
            (1100, "book_update", "bid", 100.0, 8.0),
            (1200, "trade", "buy", 101.0, 1.0),
        ],
    )

    result = validate_parquet(path, load=True)

    assert result.ok
    assert result.rows == 4
    assert result.errors == []


def test_validate_parquet_reports_contract_errors(tmp_path):
    path = _write_events(
        tmp_path,
        [
            (1000, "book_level", "bid", 100.0, 5.0),
            (900, "trade", "bid", -1.0, 0.0),
        ],
    )

    result = validate_parquet(path)

    assert not result.ok
    assert {
        "UNSORTED_TIMESTAMP",
        "INVALID_SIDE",
        "INVALID_PRICE",
        "INVALID_QUANTITY",
    }.issubset(_codes(result))


def test_validate_parquet_update_only_warns_but_is_valid(tmp_path):
    path = _write_events(tmp_path, [(1000, "book_update", "bid", 100.0, 5.0)])

    result = validate_parquet(path)

    assert result.ok
    assert [warning.code for warning in result.warnings] == ["NO_BOOK_LEVEL"]


def test_validate_parquet_detects_duplicate_snapshot_levels(tmp_path):
    path = _write_events(
        tmp_path,
        [
            (1000, "book_level", "bid", 100.0, 5.0),
            (1000, "book_level", "bid", 100.0, 7.0),
        ],
    )

    result = validate_parquet(path)

    assert not result.ok
    assert "DUPLICATE_BOOK_LEVEL" in _codes(result)


def test_cli_validate_success(tmp_path, capsys):
    path = _write_events(
        tmp_path,
        [
            (1000, "book_level", "bid", 100.0, 5.0),
            (1000, "book_level", "ask", 101.0, 3.0),
        ],
    )

    exit_code = cli_main(["validate", str(path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "OK:" in output


def test_cli_validate_failure(tmp_path, capsys):
    path = _write_events(tmp_path, [(1000, "trade", "bid", 100.0, 1.0)])

    exit_code = cli_main(["validate", str(path)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "ERROR:" in output
    assert "INVALID_SIDE" in output
