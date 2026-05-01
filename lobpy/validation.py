"""Validation helpers for lobpy parquet event files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REQUIRED_COLUMNS = ("timestamp", "event_type", "side", "price", "quantity")
OPTIONAL_COLUMNS = ("exchange_timestamp", "exchange", "symbol", "sequence")
CONTRACT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

EVENT_TYPES = {"book_level", "book_update", "trade"}
LOB_EVENT_TYPES = {"book_level", "book_update"}
LOB_SIDES = {"bid", "ask"}
TRADE_SIDES = {"buy", "sell"}
EVENT_ORDER = {"book_level": 0, "book_update": 1, "trade": 2}


@dataclass
class ValidationIssue:
    """A single parquet validation issue."""

    severity: str
    code: str
    message: str
    rows: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.rows:
            data["rows"] = self.rows
        return data


@dataclass
class ParquetValidationResult:
    """Result returned by :func:`validate_parquet`."""

    path: str
    rows: Optional[int] = None
    columns: list[str] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    elapsed: Optional[float] = None

    @property
    def ok(self) -> bool:
        """Return ``True`` when no validation errors were found."""
        return not self.errors

    def add_error(self, code: str, message: str, rows: Optional[Iterable[int]] = None) -> None:
        self.errors.append(ValidationIssue("error", code, message, _row_list(rows)))

    def add_warning(self, code: str, message: str, rows: Optional[Iterable[int]] = None) -> None:
        self.warnings.append(ValidationIssue("warning", code, message, _row_list(rows)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data: dict[str, Any] = {
            "path": self.path,
            "ok": self.ok,
            "rows": self.rows,
            "columns": self.columns,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }
        if self.elapsed is not None:
            data["elapsed"] = round(self.elapsed, 3)
        return data


@dataclass(frozen=True)
class _TypeSpec:
    label: str
    predicate: str
    strict_predicate: Optional[str] = None


_TYPE_SPECS = {
    "timestamp": _TypeSpec("integer timestamp", "integer"),
    "exchange_timestamp": _TypeSpec("integer exchange timestamp", "integer"),
    "event_type": _TypeSpec("string", "string"),
    "side": _TypeSpec("string", "string"),
    "exchange": _TypeSpec("string", "string"),
    "symbol": _TypeSpec("string", "string"),
    "sequence": _TypeSpec("string or integer", "string_or_integer"),
    "price": _TypeSpec("numeric price", "numeric", "floating"),
    "quantity": _TypeSpec("numeric quantity", "numeric", "floating"),
}


def validate_parquet(
    path: str | Path,
    *,
    strict: bool = False,
    load: bool = False,
    full_check: bool = False,
    max_rows: int = 5,
) -> ParquetValidationResult:
    """Validate a lobpy parquet event file.

    The default validator checks the columns used by ``TL.from_parquet`` and
    the invariants documented in ``docs/LLM.md``:

    - required columns are present: ``timestamp``, ``event_type``, ``side``,
      ``price``, ``quantity``;
    - no nulls in known contract columns;
    - event type, side, numeric range, ordering, and snapshot uniqueness rules;
    - optional metadata columns are validated when present.

    Args:
        path: Parquet file path.
        strict: If ``True``, require the full documented schema (including
            optional metadata columns) and exact floating types for price and
            quantity.
        load: If ``True``, additionally try ``TL.from_parquet(path, mode='lazy')``
            after structural validation.
        full_check: If ``True``, replay all events in eager mode and run
            ``lob.check()`` after every timestamp to verify the order book
            never becomes crossed or inconsistent.
        max_rows: Maximum number of row indices attached to each issue.

    Returns:
        A :class:`ParquetValidationResult`.
    """

    import time as _time

    t0 = _time.perf_counter()

    path_obj = Path(path)
    result = ParquetValidationResult(str(path_obj))

    try:
        if not path_obj.exists():
            result.add_error("FILE_NOT_FOUND", f"File does not exist: {path_obj}")
            return result
        if not path_obj.is_file():
            result.add_error("NOT_A_FILE", f"Path is not a file: {path_obj}")
            return result

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:  # pragma: no cover - depends on optional env
            result.add_error(
                "MISSING_DEPENDENCY",
                "pyarrow is required to validate parquet files. "
                "Install with: pip install lobpy[export]",
            )
            return result

        try:
            parquet_file = pq.ParquetFile(path_obj)
            schema = parquet_file.schema_arrow
        except Exception as exc:
            result.add_error("READ_ERROR", f"Could not read parquet file: {exc}")
            return result

        result.columns = list(schema.names)
        names = set(schema.names)

        missing_required = [col for col in REQUIRED_COLUMNS if col not in names]
        for col in missing_required:
            result.add_error("MISSING_COLUMN", f"Missing required column: {col}")

        if strict:
            for col in OPTIONAL_COLUMNS:
                if col not in names:
                    result.add_error("MISSING_COLUMN", f"Missing strict-schema column: {col}")

        _check_schema_types(result, schema, strict=strict, pa=pa)

        if missing_required:
            return result

        read_columns = [col for col in CONTRACT_COLUMNS if col in names]
        try:
            table = pq.read_table(path_obj, columns=read_columns)
        except Exception as exc:
            result.add_error("READ_ERROR", f"Could not read parquet table: {exc}")
            return result

        result.rows = table.num_rows
        if table.num_rows == 0:
            result.add_error("EMPTY_FILE", "Parquet file contains no rows")
            return result

        try:
            df = table.to_pandas()
        except Exception as exc:
            result.add_error("PANDAS_CONVERSION_ERROR", f"Could not convert parquet data: {exc}")
            return result

        _check_nulls(result, df, max_rows=max_rows)
        _check_values(result, df, max_rows=max_rows)
        _check_ordering(result, df, max_rows=max_rows)
        _check_uniqueness(result, df, max_rows=max_rows)
        _check_snapshot_presence(result, df, max_rows=max_rows)

        if load and result.ok:
            _check_lobpy_load(result, path_obj)

        if full_check and result.ok:
            _check_lobpy_full(result, path_obj, max_rows=max_rows)

        return result
    finally:
        result.elapsed = _time.perf_counter() - t0


def _row_list(rows: Optional[Iterable[int]]) -> list[int]:
    if rows is None:
        return []
    return [int(row) for row in rows]


def _sample_rows(mask: Any, *, max_rows: int) -> list[int]:
    values = np.asarray(mask, dtype=bool)
    return np.flatnonzero(values)[:max_rows].astype(int).tolist()


def _type_matches(kind: str, arrow_type: Any, pa: Any) -> bool:
    types = pa.types
    if kind == "integer":
        return types.is_integer(arrow_type)
    if kind == "string":
        return types.is_string(arrow_type) or types.is_large_string(arrow_type)
    if kind == "numeric":
        return types.is_integer(arrow_type) or types.is_floating(arrow_type)
    if kind == "floating":
        return types.is_floating(arrow_type)
    if kind == "string_or_integer":
        return _type_matches("string", arrow_type, pa) or _type_matches("integer", arrow_type, pa)
    return False


def _check_schema_types(
    result: ParquetValidationResult,
    schema: Any,
    *,
    strict: bool,
    pa: Any,
) -> None:
    for schema_field in schema:
        if schema_field.name not in _TYPE_SPECS:
            continue
        spec = _TYPE_SPECS[schema_field.name]
        if strict and spec.strict_predicate is not None:
            predicate = spec.strict_predicate
        else:
            predicate = spec.predicate
        if not _type_matches(predicate, schema_field.type, pa):
            result.add_error(
                "INVALID_COLUMN_TYPE",
                f"Column '{schema_field.name}' has type {schema_field.type}; expected {spec.label}",
            )
        if strict and schema_field.nullable:
            result.add_error(
                "NULLABLE_COLUMN",
                f"Column '{schema_field.name}' is nullable in schema; "
                "strict schema requires non-nullable",
            )


def _check_nulls(result: ParquetValidationResult, df: Any, *, max_rows: int) -> None:
    for col in [col for col in CONTRACT_COLUMNS if col in df.columns]:
        mask = df[col].isna()
        if bool(mask.any()):
            count = int(mask.sum())
            result.add_error(
                "NULL_VALUE",
                f"Column '{col}' contains {count} null value(s)",
                _sample_rows(mask, max_rows=max_rows),
            )


def _check_values(result: ParquetValidationResult, df: Any, *, max_rows: int) -> None:
    event_type = df["event_type"]
    side = df["side"]
    price = df["price"]
    quantity = df["quantity"]
    timestamp = df["timestamp"]

    invalid_event = ~event_type.isin(EVENT_TYPES)
    if bool(invalid_event.any()):
        result.add_error(
            "INVALID_EVENT_TYPE",
            "event_type must be one of: book_level, book_update, trade",
            _sample_rows(invalid_event, max_rows=max_rows),
        )

    lob_event = event_type.isin(LOB_EVENT_TYPES)
    trade_event = event_type == "trade"

    invalid_lob_side = lob_event & ~side.isin(LOB_SIDES)
    if bool(invalid_lob_side.any()):
        result.add_error(
            "INVALID_SIDE",
            "book_level/book_update rows must use side 'bid' or 'ask'",
            _sample_rows(invalid_lob_side, max_rows=max_rows),
        )

    invalid_trade_side = trade_event & ~side.isin(TRADE_SIDES)
    if bool(invalid_trade_side.any()):
        result.add_error(
            "INVALID_SIDE",
            "trade rows must use side 'buy' or 'sell'",
            _sample_rows(invalid_trade_side, max_rows=max_rows),
        )

    bad_price = price <= 0
    if bool(bad_price.any()):
        result.add_error(
            "INVALID_PRICE",
            "price must be strictly positive (> 0)",
            _sample_rows(bad_price, max_rows=max_rows),
        )

    bad_update_qty = (event_type == "book_update") & (quantity < 0)
    if bool(bad_update_qty.any()):
        result.add_error(
            "INVALID_QUANTITY",
            "book_update quantity must be non-negative (>= 0)",
            _sample_rows(bad_update_qty, max_rows=max_rows),
        )

    bad_positive_qty = event_type.isin({"book_level", "trade"}) & (quantity <= 0)
    if bool(bad_positive_qty.any()):
        result.add_error(
            "INVALID_QUANTITY",
            "book_level and trade quantity must be strictly positive (> 0)",
            _sample_rows(bad_positive_qty, max_rows=max_rows),
        )

    bad_timestamp = timestamp < 0
    if bool(bad_timestamp.any()):
        result.add_error(
            "INVALID_TIMESTAMP",
            "timestamp must be non-negative (>= 0)",
            _sample_rows(bad_timestamp, max_rows=max_rows),
        )

    if "exchange_timestamp" in df.columns:
        bad_exchange_timestamp = df["exchange_timestamp"] < 0
        if bool(bad_exchange_timestamp.any()):
            result.add_error(
                "INVALID_EXCHANGE_TIMESTAMP",
                "exchange_timestamp must be non-negative (>= 0)",
                _sample_rows(bad_exchange_timestamp, max_rows=max_rows),
            )

    if "sequence" in df.columns and not _sequence_values_are_lobpy_compatible(df["sequence"]):
        result.add_warning(
            "SEQUENCE_NOT_NUMERIC",
            "sequence is documented as metadata, but current TL.from_parquet stores it as int64; "
            "non-numeric sequence values will fail --load",
        )


def _check_ordering(result: ParquetValidationResult, df: Any, *, max_rows: int) -> None:
    timestamp = df["timestamp"]
    timestamp_decrease = timestamp.diff() < 0
    if bool(timestamp_decrease.any()):
        result.add_error(
            "UNSORTED_TIMESTAMP",
            "Rows must be sorted by timestamp ascending",
            _sample_rows(timestamp_decrease, max_rows=max_rows),
        )

    order_codes = df["event_type"].map(EVENT_ORDER)
    same_timestamp = timestamp.eq(timestamp.shift())
    order_decrease = same_timestamp & (order_codes < order_codes.shift())
    if bool(order_decrease.any()):
        result.add_error(
            "INVALID_EVENT_ORDER",
            "Within the same timestamp, rows must be ordered: book_level -> book_update -> trade",
            _sample_rows(order_decrease, max_rows=max_rows),
        )


def _check_uniqueness(result: ParquetValidationResult, df: Any, *, max_rows: int) -> None:
    levels = df[df["event_type"] == "book_level"]
    if levels.empty:
        return

    duplicates = levels.duplicated(subset=["timestamp", "side", "price"], keep=False)
    if bool(duplicates.any()):
        rows = levels.index[duplicates].to_numpy()[:max_rows].astype(int).tolist()
        result.add_error(
            "DUPLICATE_BOOK_LEVEL",
            "Within a book_level batch, (timestamp, side, price) must be unique",
            rows,
        )


def _check_snapshot_presence(result: ParquetValidationResult, df: Any, *, max_rows: int) -> None:
    has_book_level = bool((df["event_type"] == "book_level").any())
    if not has_book_level:
        result.add_warning(
            "NO_BOOK_LEVEL",
            "No book_level snapshot rows found; lobpy can parse this as an initially empty book",
        )
        return

    lob_rows = df[df["event_type"].isin(LOB_EVENT_TYPES)]
    if lob_rows.empty:
        return
    first_lob_idx = int(lob_rows.index[0])
    if df.loc[first_lob_idx, "event_type"] == "book_update":
        result.add_warning(
            "BOOK_UPDATE_BEFORE_SNAPSHOT",
            "First LOB event is a book_update; lobpy will apply it to an initially empty book",
            [first_lob_idx],
        )


def _sequence_values_are_lobpy_compatible(series: Any) -> bool:
    if series.empty:
        return True
    try:
        series.astype("int64")
    except Exception:
        return False
    return True


def _check_lobpy_load(result: ParquetValidationResult, path: Path) -> None:
    try:
        from .tl import TL

        tl = TL()
        tl.from_parquet(path, mode="lazy")
    except Exception as exc:
        result.add_error("LOBPY_LOAD_FAILED", f"TL.from_parquet(..., mode='lazy') failed: {exc}")


def _check_lobpy_full(
    result: ParquetValidationResult, path: Path, *, max_rows: int = 5
) -> None:
    """Replay events in C and run lob.check() after each timestamp."""
    try:
        import pyarrow.parquet as pq
        import numpy as np

        from lobpy._cext import ffi, lib

        # Read only the 5 contract columns — rest are irrelevant for the check
        table = pq.read_table(
            path, columns=["timestamp", "event_type", "side", "price", "quantity"]
        )
        n_rows = table.num_rows
        if n_rows == 0:
            return

        # --- encode string columns to uint8 (event_type / side guaranteed valid by step 5) ---

        et = table.column("event_type").to_numpy()
        event_types = np.where(et == "book_update", 1, np.where(et == "trade", 2, 0)).astype(np.uint8)

        s = table.column("side").to_numpy()
        sides = np.where((s == "ask") | (s == "sell"), 1, 0).astype(np.uint8)

        timestamps = table.column("timestamp").to_numpy().astype(np.int64)
        prices = table.column("price").to_numpy().astype(np.float64)
        quantities = table.column("quantity").to_numpy().astype(np.float64)

        # --- pre-allocate output arrays (worst case: every timestamp fails) ---

        out_failed_ts = np.empty(n_rows, dtype=np.int64)
        out_failed_bid = np.empty(n_rows, dtype=np.float64)
        out_failed_ask = np.empty(n_rows, dtype=np.float64)
        n_failed_p = ffi.new("int *")

        def _p(arr, ct):
            return ffi.cast(ct, arr.ctypes.data)

        lib.lobpy_validate_full(
            _p(timestamps, "const long long *"),
            _p(event_types, "const uint8_t *"),
            _p(sides, "const uint8_t *"),
            _p(prices, "const double *"),
            _p(quantities, "const double *"),
            n_rows,
            _p(out_failed_ts, "long long *"),
            _p(out_failed_bid, "double *"),
            _p(out_failed_ask, "double *"),
            n_failed_p,
        )

        n_failed = n_failed_p[0]
        if n_failed < 0:
            result.add_error(
                "LOBPY_FULL_CHECK_ERROR",
                "Full check: C book allocation failed",
            )
        elif n_failed > 0:
            import math

            lines: list[str] = []
            for i in range(min(n_failed, max_rows)):
                ts = int(out_failed_ts[i])
                bid = out_failed_bid[i]
                ask = out_failed_ask[i]
                bid_s = f"{bid:.4f}" if not math.isnan(bid) else "nan"
                ask_s = f"{ask:.4f}" if not math.isnan(ask) else "nan"
                lines.append(f"  ts={ts}  best_bid={bid_s}  best_ask={ask_s}")
            if n_failed > max_rows:
                lines.append(f"  ... ({n_failed - max_rows} more omitted)")
            result.add_error(
                "LOBPY_FULL_CHECK_FAILED",
                f"LOB consistency check failed at {n_failed} timestamp(s):\n"
                + "\n".join(lines),
            )
    except Exception as exc:
        result.add_error(
            "LOBPY_FULL_CHECK_ERROR",
            f"Full check could not be completed: {exc}",
        )
