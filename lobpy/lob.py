import time
from typing import Any

import numpy as np

from lobpy._cext import ffi, lib


def _normalize_side(side: str) -> str:
    if side in ("b", "bid"):
        return "bid"
    elif side in ("a", "ask"):
        return "ask"
    else:
        return side


def _make_bid_array(levels) -> np.ndarray:
    if not len(levels):
        return np.empty((0, 2), dtype=np.float64)
    arr = np.array(levels, dtype=np.float64)
    return arr[np.argsort(-arr[:, 0])]


def _make_ask_array(levels) -> np.ndarray:
    if not len(levels):
        return np.empty((0, 2), dtype=np.float64)
    arr = np.array(levels, dtype=np.float64)
    return arr[np.argsort(arr[:, 0])]


def _levels_to_arrays(levels):
    """Return (prices, qtys) as contiguous float64 arrays.  Caller must keep
    the arrays alive for as long as any C pointer into them is in use."""
    if not levels:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    p_arr = np.array([float(p) for p, _ in levels], dtype=np.float64)
    q_arr = np.array([float(q) for _, q in levels], dtype=np.float64)
    return p_arr, q_arr


def _get_sidebook_data(lob_ptr, getter_fn):
    data_pp = ffi.new("const double **")
    len_p = ffi.new("int *")
    getter_fn(lob_ptr, data_pp, len_p)
    n = len_p[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    buf = ffi.buffer(ffi.cast("double *", data_pp[0]), n * 2 * 8)
    flat = np.frombuffer(buf, dtype=np.float64).copy()
    return flat.reshape(n, 2)


class _TopValueNumericMixin:
    def __getitem__(self, index):
        raise NotImplementedError

    def _top(self) -> float:
        return float(self[0])

    def __float__(self) -> float:
        return self._top()

    def __int__(self) -> int:
        return int(self._top())

    def __add__(self, other: Any):
        return self._top() + float(other)

    def __radd__(self, other: Any):
        return float(other) + self._top()

    def __sub__(self, other: Any):
        return self._top() - float(other)

    def __rsub__(self, other: Any):
        return float(other) - self._top()

    def __mul__(self, other: Any):
        return self._top() * float(other)

    def __rmul__(self, other: Any):
        return float(other) * self._top()

    def __truediv__(self, other: Any):
        return self._top() / float(other)

    def __rtruediv__(self, other: Any):
        return float(other) / self._top()

    def __neg__(self):
        return -self._top()

    def __abs__(self):
        return abs(self._top())

    def __lt__(self, other: Any) -> bool:
        return self._top() < float(other)

    def __le__(self, other: Any) -> bool:
        return self._top() <= float(other)

    def __gt__(self, other: Any) -> bool:
        return self._top() > float(other)

    def __ge__(self, other: Any) -> bool:
        return self._top() >= float(other)


class PriceAccessor(_TopValueNumericMixin):
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, index):
        if index < len(self._arr):
            return float(self._arr[index, 0])
        return 0.0

    def __eq__(self, other):
        if isinstance(other, PriceAccessor):
            return np.array_equal(self._arr, other._arr)
        return self[0] == other

    def __repr__(self):
        return str(self[0])


class QuantityAccessor(_TopValueNumericMixin):
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, index):
        if index < len(self._arr):
            return float(self._arr[index, 1])
        return 0.0

    def __eq__(self, other):
        if isinstance(other, QuantityAccessor):
            return np.array_equal(self._arr, other._arr)
        return self[0] == other

    def __repr__(self):
        return str(self[0])


class VolumeImbalanceAccessor(_TopValueNumericMixin):
    def __init__(self, bids: np.ndarray, asks: np.ndarray):
        self._bids = bids
        self._asks = asks

    def __getitem__(self, index):
        nlevels = index + 1
        total_bid = float(self._bids[:nlevels, 1].sum()) if len(self._bids) > 0 else 0.0
        total_ask = float(self._asks[:nlevels, 1].sum()) if len(self._asks) > 0 else 0.0
        if total_bid + total_ask == 0:
            return 0.0
        return (total_bid - total_ask) / (total_bid + total_ask)

    def __eq__(self, other):
        return self[0] == other

    def __repr__(self):
        return str(self[0])


class LOB:
    def __init__(self, name=None, tick_size=1, *, bids=None, asks=None) -> None:
        if name is None:
            name = f"lob{id(self)}"
        self._tick_size = float(tick_size)
        self._name = name
        ts = int(time.time() * 1000)
        self._ptr = lib.lob_create(name.encode(), float(tick_size), ts)
        self._owns_ptr = True
        self._bids_cache = None
        self._asks_cache = None
        if bids or asks:
            self.set_snapshot(bids or [], asks or [])

    @classmethod
    def _from_ptr(cls, ptr):
        """Wrap an existing LobBook* without taking ownership.

        The caller (usually LOBts) retains ownership and is responsible for
        keeping the underlying LobBook alive for as long as this wrapper is used.
        """
        obj = cls.__new__(cls)
        obj._ptr = ptr
        obj._owns_ptr = False
        obj._bids_cache = None
        obj._asks_cache = None
        obj._tick_size = float(lib.lob_get_tick_size(ptr))
        obj._name = ffi.string(lib.lob_name(ptr)).decode()
        return obj

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr and getattr(self, "_owns_ptr", True):
            lib.lob_destroy(self._ptr)

    def _invalidate_cache(self):
        self._bids_cache = None
        self._asks_cache = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def tick_size(self):
        return self._tick_size

    @tick_size.setter
    def tick_size(self, value):
        self._tick_size = float(value)
        lib.lob_set_tick_size(self._ptr, float(value))

    def _set_tick_size(self, tick_size) -> None:
        self.tick_size = tick_size

    @property
    def timestamp(self):
        return lib.lob_timestamp(self._ptr)

    @timestamp.setter
    def timestamp(self, value):
        lib.lob_set_timestamp(self._ptr, int(value))

    @property
    def _bids(self) -> np.ndarray:
        if self._bids_cache is None:
            self._bids_cache = _get_sidebook_data(self._ptr, lib.lob_get_bids)
        return self._bids_cache

    @_bids.setter
    def _bids(self, value):
        self._bids_cache = value

    @property
    def _asks(self) -> np.ndarray:
        if self._asks_cache is None:
            self._asks_cache = _get_sidebook_data(self._ptr, lib.lob_get_asks)
        return self._asks_cache

    @_asks.setter
    def _asks(self, value):
        self._asks_cache = value

    def set_snapshot(self, bids, asks, timestamp=0):
        b_prices, b_qtys = _levels_to_arrays(bids)
        a_prices, a_qtys = _levels_to_arrays(asks)
        bn = len(b_prices)
        an = len(a_prices)
        bp = ffi.cast("double *", b_prices.ctypes.data) if bn else ffi.NULL
        bq = ffi.cast("double *", b_qtys.ctypes.data) if bn else ffi.NULL
        ap = ffi.cast("double *", a_prices.ctypes.data) if an else ffi.NULL
        aq = ffi.cast("double *", a_qtys.ctypes.data) if an else ffi.NULL
        lib.lob_set_snapshot(self._ptr, int(timestamp), bp, bq, bn, ap, aq, an)
        self._invalidate_cache()

    def set_updates(self, updates, timestamp=0):
        if not updates:
            return
        n = len(updates)
        sides = np.empty(n, dtype=np.uint8)
        prices = np.empty(n, dtype=np.float64)
        qtys = np.empty(n, dtype=np.float64)
        for i, (side, price, size) in enumerate(updates):
            s = _normalize_side(side)
            sides[i] = 0 if s == "bid" else 1
            prices[i] = float(price)
            qtys[i] = float(size)
        s_ptr = ffi.cast("uint8_t *", sides.ctypes.data)
        p_ptr = ffi.cast("double *", prices.ctypes.data)
        q_ptr = ffi.cast("double *", qtys.ctypes.data)
        lib.lob_set_updates(self._ptr, int(timestamp), s_ptr, p_ptr, q_ptr, n)
        self._invalidate_cache()

    def _delete_level(self, side, price_level, timestamp=0):
        side = _normalize_side(side)
        if side == "bid":
            self._delete_bid_level(price_level, timestamp)
        elif side == "ask":
            self._delete_ask_level(price_level, timestamp)

    def _delete_ask_level(self, price_level, timestamp=0):
        lib.lob_delete_ask(self._ptr, int(timestamp), float(price_level))
        self._invalidate_cache()

    def _delete_bid_level(self, price_level, timestamp=0):
        lib.lob_delete_bid(self._ptr, int(timestamp), float(price_level))
        self._invalidate_cache()

    def at(self, side, price) -> float:
        s = _normalize_side(side)
        return lib.lob_at(self._ptr, 0 if s == "bid" else 1, float(price))

    def update(self, side, price_level, size, timestamp=0):
        if size == 0:
            self._delete_level(side, price_level, timestamp)
            return
        s = _normalize_side(side)
        lib.lob_update_level(
            self._ptr, int(timestamp), 0 if s == "bid" else 1, float(price_level), float(size)
        )
        self._invalidate_cache()

    def _update_bid(self, price_level, size, timestamp=0):
        if size == 0:
            self._delete_bid_level(price_level, timestamp)
            return
        lib.lob_update_level(self._ptr, int(timestamp), 0, float(price_level), float(size))
        self._invalidate_cache()

    def _update_ask(self, price_level, size, timestamp=0):
        if size == 0:
            self._delete_ask_level(price_level, timestamp)
            return
        lib.lob_update_level(self._ptr, int(timestamp), 1, float(price_level), float(size))
        self._invalidate_cache()

    @property
    def ask(self):
        return PriceAccessor(self._asks)

    @property
    def bid(self):
        return PriceAccessor(self._bids)

    @property
    def spread(self):
        return lib.lob_spread(self._ptr)

    @property
    def spread_tick(self):
        return lib.lob_spread_tick(self._ptr)

    @property
    def spread_rel(self):
        return lib.lob_spread_rel(self._ptr)

    @property
    def midprice(self):
        return lib.lob_midprice(self._ptr)

    @property
    def vw_midprice(self):
        return lib.lob_vw_midprice(self._ptr)

    @property
    def bidq(self):
        return QuantityAccessor(self._bids)

    @property
    def askq(self):
        return QuantityAccessor(self._asks)

    @property
    def vi(self):
        return VolumeImbalanceAccessor(self._bids, self._asks)

    def get_delta(self, bids, asks, timestamp=0):
        bid_deltas = []
        ask_deltas = []

        old_bids = dict(self._bids)
        old_asks = dict(self._asks)

        new_bids = {price: qty for price, qty in bids}
        new_asks = {price: qty for price, qty in asks}

        for price, quantity in new_bids.items():
            if quantity != old_bids.get(price, 0.0):
                bid_deltas.append((price, quantity))

        for price in old_bids:
            if price not in new_bids:
                bid_deltas.append((price, 0.0))

        for price, quantity in new_asks.items():
            if quantity != old_asks.get(price, 0.0):
                ask_deltas.append((price, quantity))

        for price in old_asks:
            if price not in new_asks:
                ask_deltas.append((price, 0.0))

        b_prices, b_qtys = _levels_to_arrays(bids)
        a_prices, a_qtys = _levels_to_arrays(asks)
        bn = len(b_prices)
        an = len(a_prices)
        bp = ffi.cast("double *", b_prices.ctypes.data) if bn else ffi.NULL
        bq = ffi.cast("double *", b_qtys.ctypes.data) if bn else ffi.NULL
        ap = ffi.cast("double *", a_prices.ctypes.data) if an else ffi.NULL
        aq = ffi.cast("double *", a_qtys.ctypes.data) if an else ffi.NULL
        lib.lob_set_snapshot(self._ptr, int(timestamp), bp, bq, bn, ap, aq, an)
        self._invalidate_cache()

        return (bid_deltas, ask_deltas)

    def to_np(self, side=None, nlevels=None):
        if side == "b":
            arr = self._bids[:nlevels] if nlevels is not None else self._bids
            return arr.copy() if len(arr) > 0 else np.empty((0, 2))
        elif side == "a":
            arr = self._asks[:nlevels] if nlevels is not None else self._asks
            return arr.copy() if len(arr) > 0 else np.empty((0, 2))
        else:
            if nlevels is not None:
                bid_n = (nlevels + 1) // 2
                ask_n = nlevels // 2
            else:
                bid_n = len(self._bids)
                ask_n = len(self._asks)
            bid_levels = self._bids[:bid_n]
            ask_levels = self._asks[:ask_n]

            if len(bid_levels) == 0 and len(ask_levels) == 0:
                return np.empty((0, 3), dtype=object)

            data = []
            for price, size in bid_levels:
                data.append(("b", price, size))
            for price, size in ask_levels:
                data.append(("a", price, size))
            return np.array(data, dtype=object)

    def to_pd(self, side=None, nlevels=None):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pd() method. "
                "Install it with: pip install pandas[export]"
            ) from e

        if side == "b":
            arr = self._bids[:nlevels] if nlevels is not None else self._bids
            return pd.DataFrame(arr, columns=["price", "size"])
        elif side == "a":
            arr = self._asks[:nlevels] if nlevels is not None else self._asks
            return pd.DataFrame(arr, columns=["price", "size"])
        else:
            if nlevels is not None:
                bid_n = (nlevels + 1) // 2
                ask_n = nlevels // 2
            else:
                bid_n = len(self._bids)
                ask_n = len(self._asks)
            bid_levels = self._bids[:bid_n]
            ask_levels = self._asks[:ask_n]

            data = []
            for price, size in bid_levels:
                data.append((price, size, "b"))
            for price, size in ask_levels:
                data.append((price, size, "a"))
            return pd.DataFrame(data, columns=["price", "size", "side"])

    def to_csv(self, path, side=None, nlevels=None):
        df = self.to_pd(side, nlevels)
        df.to_csv(path, index=False)

    def to_xlsx(self, path, side=None, nlevels=None):
        df = self.to_pd(side, nlevels)
        df.to_excel(path, index=False, engine="openpyxl")

    def to_parquet(self, path, side=None, nlevels=None):
        df = self.to_pd(side, nlevels)
        df.to_parquet(path, engine="pyarrow")

    def check(self):
        return bool(lib.lob_check(self._ptr))

    def get_slippage(self, volume, side="midprice"):
        if volume <= 0:
            return 0.0
        if side == "midprice":
            return 0.0
        side = _normalize_side(side)
        if side == "ask":
            return lib.lob_slippage(self._ptr, float(volume), 1)
        elif side == "bid":
            return lib.lob_slippage(self._ptr, float(volume), 0)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'midprice', 'a'/'ask', or 'b'/'bid'.")

    def len_in_tick(self, side, price):
        side = _normalize_side(side)
        if side == "bid":
            result = lib.lob_len_in_tick(self._ptr, 0, float(price))
            return result if result >= 0 else float("inf")
        elif side == "ask":
            result = lib.lob_len_in_tick(self._ptr, 1, float(price))
            return result if result >= 0 else float("inf")
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'b'/'bid' or 'a'/'ask'.")

    def diff(self, other):
        updates = []

        self_bids = dict(self._bids)
        self_asks = dict(self._asks)
        other_bids = dict(other._bids)
        other_asks = dict(other._asks)

        for price, size in other_bids.items():
            if self_bids.get(price) != size:
                updates.append(("bid", price, size))

        for price in self_bids:
            if price not in other_bids:
                updates.append(("bid", price, 0))

        for price, size in other_asks.items():
            if self_asks.get(price) != size:
                updates.append(("ask", price, size))

        for price in self_asks:
            if price not in other_asks:
                updates.append(("ask", price, 0))

        return updates

    def aggq(self, side, nlevel=None, ticks=None, price=None):
        side = _normalize_side(side)
        if side not in ("bid", "ask"):
            raise ValueError(f"Invalid side: {side}. Must be 'b'/'bid' or 'a'/'ask'.")

        s = 0 if side == "bid" else 1

        if nlevel is not None:
            return lib.lob_aggq_nlevel(self._ptr, s, nlevel)
        elif ticks is not None:
            return lib.lob_aggq_ticks(self._ptr, s, ticks)
        elif price is not None:
            return lib.lob_aggq_price(self._ptr, s, float(price))
        else:
            raise ValueError("Must specify one of: nlevel, ticks, or price")

    def __repr__(self) -> str:
        return f"<Book[{self.name}]>"
