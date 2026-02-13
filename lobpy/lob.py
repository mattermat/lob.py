import time

from .sorteddict import SortedDict

def neg(x: int) -> int:
    return -x

def _normalize_side(side: str) -> str:
    """Normalize side parameter to long form ('bid' or 'ask')."""
    if side in ('b', 'bid'):
        return 'bid'
    elif side in ('a', 'ask'):
        return 'ask'
    else:
        return side

class PriceAccessor:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, index):
        items = list(self._data.items())
        if index < len(items):
            return items[index][0]
        return .0

    def __eq__(self, other):
        if isinstance(other, PriceAccessor):
            return self._data == other._data
        return self[0] == other

    def __repr__(self):
        return str(self[0])

class QuantityAccessor:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, index):
        items = list(self._data.items())
        if index < len(items):
            return items[index][1]
        return .0

    def __eq__(self, other):
        if isinstance(other, QuantityAccessor):
            return self._data == other._data
        return self[0] == other

    def __repr__(self):
        return str(self[0])

class VolumeImbalanceAccessor:
    def __init__(self, bids, asks):
        self._bids = bids
        self._asks = asks

    def __getitem__(self, index):
        bid_items = list(self._bids.items())
        ask_items = list(self._asks.items())
        
        nlevels = index + 1
        
        total_bid = sum(size for _, size in bid_items[:nlevels])
        total_ask = sum(size for _, size in ask_items[:nlevels])
        
        if total_bid + total_ask == 0:
            return 0.0
        
        return (total_bid - total_ask) / (total_bid + total_ask)

    def __eq__(self, other):
        return self[0] == other

    def __repr__(self):
        return str(self[0])

def _get_levels(data, nlevels=None):
    """Helper to get levels from SortedDict, optionally limiting nlevels."""
    items = list(data.items())
    if nlevels is not None:
        items = items[:nlevels]
    return items

class LOB():

    def __init__(self, name=None, tick_size=1, *, bids=None, asks=None) -> None:
        if name is None:
            name = f"lob{id(self)}"
        self.name = name
        self._bids = SortedDict(neg)
        self._asks = SortedDict()
        self.tick_size = tick_size
        self.timestamp = int(time.time()*1000)
        self._crossing_detected = False
        
        if bids is None:
            bids = []
        if asks is None:
            asks = []
        for b in bids:
            self._bids[b[0]] = b[1]
        for a in asks:
            self._asks[a[0]] = a[1]

    def _set_tick_size(self, tick_size) -> None:
        self.tick_size = tick_size

    def set_snapshot(self, bids, asks, timestamp=0):
        """
        align the order book to a snapshot
        """
        self._bids.clear()
        self._asks.clear()
        for b in bids:
            self._bids[b[0]] = b[1]
        for a in asks:
            self._asks[a[0]] = a[1]
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
            if side == 'bid':
                if size == 0:
                    save_bids.pop(price, None)
                else:
                    save_bids[price] = size
            else:
                if size == 0:
                    save_asks.pop(price, None)
                else:
                    save_asks[price] = size
        
        self._bids.clear()
        self._asks.clear()
        for price, qty in save_bids.items():
            self._bids[price] = qty
        for price, qty in save_asks.items():
            self._asks[price] = qty
        
        if timestamp != 0:
            self.timestamp = timestamp

    def _delete_level(self, side, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp

        side = _normalize_side(side)
        if side == "bid":
            try:
                del self._bids[price_level]
            except KeyError:
                # log
                print(f"price level {price_level} not existing")

        elif side == "ask":
            try:
                del self._asks[price_level]
            except KeyError:
                # log
                print(f"price level {price_level} not existing")

    def _delete_ask_level(self, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        try:
            del self._asks[price_level]
        except KeyError:
            pass # TODO: error message
        return
    
    def _delete_bid_level(self, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        try:
            del self._bids[price_level]
        except KeyError:
            pass # TODO: error message
        return


    def update(self, side, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp

        if size == 0:
            self._delete_level(side, price_level, timestamp)
            return

        side = _normalize_side(side)
        if side == "bid":
            try:
                self._bids[price_level] = size

            except KeyError:
                # log
                print(f"price level {price_level} not existing, while updating")
        elif side == "ask":
            try:
                self._asks[price_level] = size
            except KeyError:
                # log
                print(f"price level {price_level} not existing, while updating")

    def _update_ask(self, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if size == 0:
            self._delete_ask_level(price_level)
        else:
            self._asks[price_level] = size
        return

    def _update_bid(self, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if size == 0:
            self._delete_bid_level(price_level)
        else:
            self._bids[price_level] = size
        return

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
        return float('nan')

    @property
    def spread_tick(self):
        """
        get the spread in ticks
        """
        spread = self.spread
        if spread != spread:
            return float('nan')
        return spread / self.tick_size

    @property
    def spread_rel(self):
        """
        get the spread as percentage of the bid level
        """
        bid_price = self.bid[0]
        if bid_price > 0:
            return self.spread / bid_price
        return float('nan')

    @property
    def midprice(self):
        """
        get the mid-price (bid + ask) / 2
        """
        ask_price = self.ask[0]
        bid_price = self.bid[0]
        if ask_price > 0 and bid_price > 0:
            return (bid_price + ask_price) / 2
        return float('nan')

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
        return float('nan')

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

        # Get current state
        old_bids = dict(self._bids.items())
        old_asks = dict(self._asks.items())

        # Convert new snapshot to dicts
        new_bids = {price: qty for price, qty in bids}
        new_asks = {price: qty for price, qty in asks}

        # Find bid changes
        for price, quantity in new_bids.items():
            old_quantity = old_bids.get(price, 0.0)
            if quantity != old_quantity:
                bid_deltas.append((price, quantity))

        # Find deleted bid levels
        for price in old_bids:
            if price not in new_bids:
                bid_deltas.append((price, 0.0))

        # Find ask changes
        for price, quantity in new_asks.items():
            old_quantity = old_asks.get(price, 0.0)
            if quantity != old_quantity:
                ask_deltas.append((price, quantity))

        # Find deleted ask levels
        for price in old_asks:
            if price not in new_asks:
                ask_deltas.append((price, 0.0))

        # Update internal state to new snapshot
        self._bids.clear()
        self._asks.clear()
        for price, qty in new_bids.items():
            self._bids[price] = qty
        for price, qty in new_asks.items():
            self._asks[price] = qty

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
        import numpy as np

        if side == 'b':
            levels = _get_levels(self._bids, nlevels)
            if not levels:
                return np.empty((0, 2))
            return np.array(levels, dtype=float)
        elif side == 'a':
            levels = _get_levels(self._asks, nlevels)
            if not levels:
                return np.empty((0, 2))
            return np.array(levels, dtype=float)
        else:
            if nlevels is None:
                bid_levels = _get_levels(self._bids)
                ask_levels = _get_levels(self._asks)
            else:
                bid_levels = _get_levels(self._bids)
                ask_levels = _get_levels(self._asks)
                
                bid_count = (nlevels + 1) // 2
                ask_count = nlevels // 2
                
                bid_levels = bid_levels[:bid_count]
                ask_levels = ask_levels[:ask_count]
            
            if not bid_levels and not ask_levels:
                return np.empty((0, 3), dtype=object)
            
            data = []
            for price, size in bid_levels:
                data.append(('b', price, size))
            for price, size in ask_levels:
                data.append(('a', price, size))
            
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
        """
        import pandas as pd

        if side == 'b':
            levels = _get_levels(self._bids, nlevels)
            return pd.DataFrame(levels, columns=['price', 'size'])
        elif side == 'a':
            levels = _get_levels(self._asks, nlevels)
            return pd.DataFrame(levels, columns=['price', 'size'])
        else:
            if nlevels is None:
                bid_levels = _get_levels(self._bids)
                ask_levels = _get_levels(self._asks)
            else:
                bid_levels = _get_levels(self._bids)
                ask_levels = _get_levels(self._asks)
                
                bid_count = (nlevels + 1) // 2
                ask_count = nlevels // 2
                
                bid_levels = bid_levels[:bid_count]
                ask_levels = ask_levels[:ask_count]
            
            data = []
            for price, size in bid_levels:
                data.append((price, size, 'b'))
            for price, size in ask_levels:
                data.append((price, size, 'a'))
            
            return pd.DataFrame(data, columns=['price', 'size', 'side'])

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
        df.to_excel(path, index=False, engine='openpyxl')

    def to_parquet(self, path, side=None, nlevels=None):
        """
        Export order book to Parquet file.

        Args:
            path: file path for Parquet output
            side: 'b' for bids, 'a' for asks, or None for both sides
            nlevels: number of top levels to export (default: all levels)
        """
        df = self.to_pd(side, nlevels)
        df.to_parquet(path, engine='pyarrow')

    def check(self):
        """
        Check consistency of the order book.

        Returns:
            True if the book is consistent (best bid < best ask or one side empty),
            False if the book is crossed (best bid >= best ask).
        """
        if not self._bids or not self._asks:
            return True
        bid_price = self.bid[0]
        ask_price = self.ask[0]
        return bid_price < ask_price

    def get_slippage(self, volume, side='midprice'):
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

        if side == 'midprice':
            return 0.0
        
        side = _normalize_side(side)
        if side == 'ask':
            remaining = volume
            total_cost = 0.0
            ask_items = list(self._asks.items())
            for price, size in ask_items:
                if remaining <= 0:
                    break
                take = min(remaining, size)
                total_cost += take * price
                remaining -= take
            if remaining > 0:
                return float('inf')
            avg_price = total_cost / volume
            return avg_price - self.midprice
        elif side == 'bid':
            remaining = volume
            total_cost = 0.0
            bid_items = list(self._bids.items())
            for price, size in bid_items:
                if remaining <= 0:
                    break
                take = min(remaining, size)
                total_cost += take * price
                remaining -= take
            if remaining > 0:
                return float('inf')
            avg_price = total_cost / volume
            return self.midprice - avg_price
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
        if side == 'bid':
            best_price = self.bid[0]
            if best_price <= 0:
                return float('inf')
            return int(round((best_price - price) / self.tick_size))
        elif side == 'ask':
            best_price = self.ask[0]
            if best_price <= 0:
                return float('inf')
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
        
        self_bids = dict(self._bids.items())
        self_asks = dict(self._asks.items())
        other_bids = dict(other._bids.items())
        other_asks = dict(other._asks.items())
        
        for price, size in other_bids.items():
            if self_bids.get(price) != size:
                updates.append(('bid', price, size))
        
        for price in self_bids:
            if price not in other_bids:
                updates.append(('bid', price, 0))
        
        for price, size in other_asks.items():
            if self_asks.get(price) != size:
                updates.append(('ask', price, size))
        
        for price in self_asks:
            if price not in other_asks:
                updates.append(('ask', price, 0))
        
        return updates

    def aggq(self, side, nlevel=None, ticks=None, price=None):
        """
        Aggregate order book quantities based on the specified criteria.

        Args:
            side: 'b' or 'bid' for bids, 'a' or 'ask' for asks - which side of the order book to aggregate
            nlevel: number of top levels to aggregate (e.g., nlevel=3 for top 3 levels)
            ticks: tick distance from the best price to aggregate
            price: price level to aggregate at or beyond

        Returns:
            Total aggregated quantity for the specified criteria

        Raises:
            ValueError: if side is invalid or no aggregation criterion is specified
        """
        side = _normalize_side(side)
        if side not in ('bid', 'ask'):
            raise ValueError(f"Invalid side: {side}. Must be 'b'/'bid' or 'a'/'ask'.")

        data = self._bids if side == 'bid' else self._asks
        items = list(data.items())

        if not items:
            return 0.0

        if nlevel is not None:
            levels = items[:nlevel]
            return sum(size for _, size in levels)
        elif ticks is not None:
            if side == 'bid':
                best_price = items[0][0]
                if best_price <= 0:
                    return 0.0
                min_price = best_price - ticks * self.tick_size
                levels = [(p, s) for p, s in items if p >= min_price]
            else:
                best_price = items[0][0]
                if best_price <= 0:
                    return 0.0
                max_price = best_price + ticks * self.tick_size
                levels = [(p, s) for p, s in items if p <= max_price]
            return sum(size for _, size in levels)
        elif price is not None:
            if side == 'bid':
                levels = [(p, s) for p, s in items if p >= price]
            else:
                levels = [(p, s) for p, s in items if p <= price]
            return sum(size for _, size in levels)
        else:
            raise ValueError("Must specify one of: nlevel, ticks, or price")

    def __repr__(self) -> str:
        return f"<Book[{self.name}]>"