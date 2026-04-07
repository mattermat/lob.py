import time
from typing import Any

import numpy as np


def _normalize_side(side: str) -> str:
    """Normalize side parameter to long form ('bid' or 'ask')."""
    if side in ("b", "bid"):
        return "bid"
    elif side in ("a", "ask"):
        return "ask"
    else:
        return side


def _make_bid_array(levels) -> np.ndarray:
    """Build bid array sorted descending by price from a list of (price, qty) pairs."""
    if not len(levels):
        return np.empty((0, 2), dtype=np.float64)
    arr = np.array(levels, dtype=np.float64)
    return arr[np.argsort(-arr[:, 0])]


def _make_ask_array(levels) -> np.ndarray:
    """Build ask array sorted ascending by price from a list of (price, qty) pairs."""
    if not len(levels):
        return np.empty((0, 2), dtype=np.float64)
    arr = np.array(levels, dtype=np.float64)
    return arr[np.argsort(arr[:, 0])]


class _TopValueNumericMixin:
    """Mixin that makes an accessor behave like float(self[0]) in numeric contexts."""

    def __getitem__(self, index):
        raise NotImplementedError

    def _top(self) -> float:
        return float(self[0])  # relies on __getitem__

    def __float__(self) -> float:
        return self._top()

    def __int__(self) -> int:
        return int(self._top())

    # arithmetic
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

    # unary
    def __neg__(self):
        return -self._top()

    def __abs__(self):
        return abs(self._top())

    # comparisons (optional but usually expected)
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
        self._arr = arr  # shape (n, 2), col 0 = price

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
        self._arr = arr  # shape (n, 2), col 1 = qty

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
        self.name = name
        self.tick_size = tick_size
        self.timestamp = int(time.time() * 1000)
        self._crossing_detected = False
        self._bids = _make_bid_array(bids if bids is not None else [])
        self._asks = _make_ask_array(asks if asks is not None else [])

    def _set_tick_size(self, tick_size) -> None:
        self.tick_size = tick_size

    def set_snapshot(self, bids, asks, timestamp=0):
        """
        align the order book to a snapshot
        """
        self._bids = _make_bid_array(bids)
        self._asks = _make_ask_array(asks)
        self.timestamp = timestamp

    def set_updates(self, updates, timestamp=0):
        """
        Push multiple updates to the order book at once.

        Args:
            updates: List of (side, price, size) tuples where:
                - side: 'b' or 'bid' for bids, 'a' or 'ask' for asks
                - price: price level
                - size: quantity (0 to delete level)
            timestamp: Optional timestamp for the updates

        Note:
            Updates are applied atomically - all or nothing.
        """
        save_bids = dict(self._bids)
        save_asks = dict(self._asks)

        for side, price, size in updates:
            side = _normalize_side(side)
            if side == "bid":
                if size == 0:
                    save_bids.pop(price, None)
                else:
                    save_bids[price] = size
            else:
                if size == 0:
                    save_asks.pop(price, None)
                else:
                    save_asks[price] = size

        self._bids = _make_bid_array(list(save_bids.items()))
        self._asks = _make_ask_array(list(save_asks.items()))

        if timestamp != 0:
            self.timestamp = timestamp

    def _delete_level(self, side, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp

        side = _normalize_side(side)
        if side == "bid":
            self._delete_bid_level(price_level)
        elif side == "ask":
            self._delete_ask_level(price_level)

    def _delete_ask_level(self, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if len(self._asks) == 0:
            print(f"price level {price_level} not existing on ask side")
            return
        mask = self._asks[:, 0] != price_level
        if mask.all():
            print(f"price level {price_level} not existing on ask side")
            return
        self._asks = self._asks[mask]

    def _delete_bid_level(self, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if len(self._bids) == 0:
            print(f"price level {price_level} not existing on bid side")
            return
        mask = self._bids[:, 0] != price_level
        if mask.all():
            print(f"price level {price_level} not existing on bid side")
            return
        self._bids = self._bids[mask]

    def at(self, side, price) -> float:
        """Return the quantity at a given price level, or 0 if not present."""
        if side in ("b", "bid"):
            arr = self._bids
            if len(arr) == 0:
                return 0
            idx = np.searchsorted(-arr[:, 0], -price)
            if idx < len(arr) and arr[idx, 0] == price:
                return float(arr[idx, 1])
            return 0
        else:
            arr = self._asks
            if len(arr) == 0:
                return 0
            idx = np.searchsorted(arr[:, 0], price)
            if idx < len(arr) and arr[idx, 0] == price:
                return float(arr[idx, 1])
            return 0

    def update(self, side, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp

        if size == 0:
            self._delete_level(side, price_level)
            return

        side = _normalize_side(side)
        if side == "bid":
            self._update_bid(price_level, size)
        elif side == "ask":
            self._update_ask(price_level, size)

    def _update_ask(self, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if size == 0:
            self._delete_ask_level(price_level)
            return
        if len(self._asks) == 0:
            self._asks = np.array([[price_level, size]], dtype=np.float64)
            return
        prices = self._asks[:, 0]
        idx = np.searchsorted(prices, price_level)
        if idx < len(prices) and prices[idx] == price_level:
            self._asks[idx, 1] = size
        else:
            self._asks = np.insert(self._asks, idx, [price_level, size], axis=0)

    def _update_bid(self, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if size == 0:
            self._delete_bid_level(price_level)
            return
        if len(self._bids) == 0:
            self._bids = np.array([[price_level, size]], dtype=np.float64)
            return
        prices = self._bids[:, 0]
        idx = np.searchsorted(-prices, -price_level)
        if idx < len(prices) and prices[idx] == price_level:
            self._bids[idx, 1] = size
        else:
            self._bids = np.insert(self._bids, idx, [price_level, size], axis=0)

    @property
    def ask(self):
        """
        get the best ask price
        """
        return PriceAccessor(self._asks)

    @property
    def bid(self):
        """
        get the best bid price
        """
        return PriceAccessor(self._bids)

    @property
    def spread(self):
        """
        get the bid-ask spread (ask price - bid price)
        """
        ask_price = self.ask[0]
        bid_price = self.bid[0]
        if ask_price > 0 and bid_price > 0:
            return ask_price - bid_price
        return float("nan")

    @property
    def spread_tick(self):
        """
        get the spread in ticks
        """
        spread = self.spread
        if spread != spread:
            return float("nan")
        return spread / self.tick_size

    @property
    def spread_rel(self):
        """
        get the spread as percentage of the bid level
        """
        bid_price = self.bid[0]
        if bid_price > 0:
            return self.spread / bid_price
        return float("nan")

    @property
    def midprice(self):
        """
        get the mid-price (bid + ask) / 2
        """
        ask_price = self.ask[0]
        bid_price = self.bid[0]
        if ask_price > 0 and bid_price > 0:
            return (bid_price + ask_price) / 2
        return float("nan")

    @property
    def vw_midprice(self):
        """
        get the volume-weighted mid-price
        """
        ask_price = self.ask[0]
        bid_price = self.bid[0]
        ask_size = self.askq[0]
        bid_size = self.bidq[0]
        if ask_price > 0 and bid_price > 0 and ask_size > 0 and bid_size > 0:
            total_size = ask_size + bid_size
            return (bid_price * bid_size + ask_price * ask_size) / total_size
        return float("nan")

    @property
    def bidq(self):
        """
        get the best bid size (indexable)
        """
        return QuantityAccessor(self._bids)

    @property
    def askq(self):
        """
        get the best ask size (indexable)
        """
        return QuantityAccessor(self._asks)

    @property
    def vi(self):
        """
        get the volume imbalance with indexing support
        """
        return VolumeImbalanceAccessor(self._bids, self._asks)

    def get_delta(self, bids, asks, timestamp=0):
        """
        Compare the provided snapshot with the current internal state and return deltas.

        Args:
            bids: List of (price, quantity) tuples for bid side
            asks: List of (price, quantity) tuples for ask side
            timestamp: Optional timestamp for the snapshot

        Returns:
            A tuple of (bid_deltas, ask_deltas) where each is a list of (price, quantity) tuples.
            quantity=0.0 means the level should be deleted.

        After computing deltas, updates the internal state to the new snapshot.

        Example:
            Current state: bids = [(100, 10), (99, 5)]
            New snapshot:  bids = [(100, 15), (98, 3)]
            Delta output:  [(100, 15), (99, 0.0), (98, 3)]
                          # 100 changed, 99 deleted, 98 new
        """
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

        self._bids = _make_bid_array(list(new_bids.items()))
        self._asks = _make_ask_array(list(new_asks.items()))

        if timestamp != 0:
            self.timestamp = timestamp

        return (bid_deltas, ask_deltas)

    def to_np(self, side=None, nlevels=None):
        """
        Export order book to numpy array.

        Args:
            side: 'b' for bids, 'a' for asks, or None for both sides
            nlevels: number of top levels to export (default: all levels)

        Returns:
            2D array with shape (n, 2) [price, size] when side specified
            2D array with shape (n, 3) [side, price, size] when side=None
        """
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
        """
        Export order book to pandas DataFrame.

        Args:
            side: 'b' for bids, 'a' for asks, or None for both sides
            nlevels: number of top levels to export (default: all levels)

        Returns:
            DataFrame with columns ['price', 'size'] when side specified
            DataFrame with columns ['price', 'size', 'side'] when side=None

        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd  # type: ignore
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
        """
        Export order book to CSV file.

        Args:
            path: file path for CSV output
            side: 'b' for bids, 'a' for asks, or None for both sides
            nlevels: number of top levels to export (default: all levels)
        """
        df = self.to_pd(side, nlevels)
        df.to_csv(path, index=False)

    def to_xlsx(self, path, side=None, nlevels=None):
        """
        Export order book to XLSX file.

        Args:
            path: file path for XLSX output
            side: 'b' for bids, 'a' for asks, or None for both sides
            nlevels: number of top levels to export (default: all levels)
        """
        df = self.to_pd(side, nlevels)
        df.to_excel(path, index=False, engine="openpyxl")

    def to_parquet(self, path, side=None, nlevels=None):
        """
        Export order book to Parquet file.

        Args:
            path: file path for Parquet output
            side: 'b' for bids, 'a' for asks, or None for both sides
            nlevels: number of top levels to export (default: all levels)
        """
        df = self.to_pd(side, nlevels)
        df.to_parquet(path, engine="pyarrow")

    def check(self):
        """
        Check consistency of the order book.

        Returns:
            True if the book is consistent (best bid < best ask or one side empty),
            False if the book is crossed (best bid >= best ask).
        """
        if len(self._bids) == 0 or len(self._asks) == 0:
            return True
        return float(self._bids[0, 0]) < float(self._asks[0, 0])

    def get_slippage(self, volume, side="midprice"):
        """
        Calculate the slippage from the top level.

        Args:
            volume: volume to execute
            side: 'midprice', 'a'/'ask', or 'b'/'bid'

        Returns:
            slippage in price units
        """
        if volume <= 0:
            return 0.0

        if side == "midprice":
            return 0.0

        side = _normalize_side(side)
        if side == "ask":
            remaining = volume
            total_cost = 0.0
            for price, size in self._asks:
                if remaining <= 0:
                    break
                take = min(remaining, size)
                total_cost += take * price
                remaining -= take
            if remaining > 0:
                return float("inf")
            return total_cost / volume - self.midprice
        elif side == "bid":
            remaining = volume
            total_cost = 0.0
            for price, size in self._bids:
                if remaining <= 0:
                    break
                take = min(remaining, size)
                total_cost += take * price
                remaining -= take
            if remaining > 0:
                return float("inf")
            return self.midprice - total_cost / volume
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'midprice', 'a'/'ask', or 'b'/'bid'.")

    def len_in_tick(self, side, price):
        """
        Return the number of ticks the provided price is far from the top of the book.

        Args:
            side: 'b' or 'bid' for bids, 'a' or 'ask' for asks
            price: price level to check

        Returns:
            number of ticks from the top level
        """
        side = _normalize_side(side)
        if side == "bid":
            best_price = self.bid[0]
            if best_price <= 0:
                return float("inf")
            return int(round((best_price - price) / self.tick_size))
        elif side == "ask":
            best_price = self.ask[0]
            if best_price <= 0:
                return float("inf")
            return int(round((price - best_price) / self.tick_size))
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'b'/'bid' or 'a'/'ask'.")

    def diff(self, other):
        """
        Difference between two LOB. Returns the updates needed to change self to other.

        Args:
            other: LOB object to compare against

        Returns:
            List of (side, price, size) tuples where size=0 means delete level
        """
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
        """
        Aggregate order book quantities based on the specified criteria.

        Args:
            side: 'b' or 'bid' for bids, 'a' or 'ask' for asks
                  - which side of the order book to aggregate
            nlevel: number of top levels to aggregate (e.g., nlevel=3 for top 3 levels)
            ticks: tick distance from the best price to aggregate
            price: price level to aggregate at or beyond

        Returns:
            Total aggregated quantity for the specified criteria

        Raises:
            ValueError: if side is invalid or no aggregation criterion is specified
        """
        side = _normalize_side(side)
        if side not in ("bid", "ask"):
            raise ValueError(f"Invalid side: {side}. Must be 'b'/'bid' or 'a'/'ask'.")

        arr = self._bids if side == "bid" else self._asks

        if len(arr) == 0:
            return 0.0

        if nlevel is not None:
            return float(arr[:nlevel, 1].sum())
        elif ticks is not None:
            best_price = arr[0, 0]
            if best_price <= 0:
                return 0.0
            if side == "bid":
                min_price = best_price - ticks * self.tick_size
                return float(arr[arr[:, 0] >= min_price, 1].sum())
            else:
                max_price = best_price + ticks * self.tick_size
                return float(arr[arr[:, 0] <= max_price, 1].sum())
        elif price is not None:
            if side == "bid":
                return float(arr[arr[:, 0] >= price, 1].sum())
            else:
                return float(arr[arr[:, 0] <= price, 1].sum())
        else:
            raise ValueError("Must specify one of: nlevel, ticks, or price")

    def __repr__(self) -> str:
        return f"<Book[{self.name}]>"
