"""
Time Series LOB (Limit Order Book)
"""

from .lob import LOB
from .sorteddict import SortedDict


class LOBts:
    """Time series of LOB objects indexed by timestamp."""

    def __init__(self, name=None, tick_size=1, mode='delta') -> None:
        """
        Initialize LOBts.

        Args:
            name: Optional identifier for the time series
            tick_size: Minimum price increment
            mode: 'delta' to store all snapshots, 'latest' to keep only current
        """
        if name is None:
            name = f"lobts{id(self)}"
        self.name = name
        self.tick_size = tick_size
        self._mode = mode
        self._lobs = SortedDict()
        self._timestamps = self._lobs.keys()

    @property
    def mode(self):
        """Return mode."""
        return self._mode

    @property
    def timestamps(self):
        """Return sorted timestamps."""
        return self._timestamps

    def set_snapshot(self, bids, asks, timestamp=0, force=False):
        """
        Create a LOB snapshot at the given timestamp.

        Args:
            bids: List of (price, size) tuples for bid side
            asks: List of (price, size) tuples for ask side
            timestamp: Timestamp for this snapshot
            force: If True and timestamp exists, replace it
        """
        if self._mode == 'latest':
            self._lobs.clear()

        lob = LOB(name=f"{self.name}_t{timestamp}", tick_size=self.tick_size, bids=bids, asks=asks)
        lob.timestamp = timestamp

        if timestamp in self._lobs and not force:
            raise ValueError(f"Timestamp {timestamp} already exists. Use force=True to overwrite.")
        self._lobs[timestamp] = lob

    def set_updates(self, updates, timestamp=0):
        """
        Apply updates to create a new LOB snapshot at timestamp.

        Args:
            updates: List of (side, price, size) tuples. Side can be 'bid'/'b' or 'ask'/'a'
            timestamp: Timestamp for this snapshot

        Returns:
            The new LOB object
        """
        if len(self._lobs) > 0:
            prev_timestamp = self._lobs.keys()[-1]
            prev_lob = self._lobs[prev_timestamp]
            new_lob = LOB(name=f"{self.name}_t{timestamp}", tick_size=self.tick_size)

            bids = list(prev_lob._bids.items())
            asks = list(prev_lob._asks.items())
            for b in bids:
                new_lob._bids[b[0]] = b[1]
            for a in asks:
                new_lob._asks[a[0]] = a[1]
            new_lob.timestamp = prev_timestamp

            for side, price, size in updates:
                # Convert short form side to long form
                if side == 'b':
                    side = 'bid'
                elif side == 'a':
                    side = 'ask'
                new_lob.update(side, price, size, 0)

            new_lob.timestamp = timestamp
            self._lobs[timestamp] = new_lob
        else:
            lob = LOB(name=f"{self.name}_t{timestamp}", tick_size=self.tick_size)
            for side, price, size in updates:
                # Convert short form side to long form
                if side == 'b':
                    side = 'bid'
                elif side == 'a':
                    side = 'ask'
                lob.update(side, price, size, 0)
            lob.timestamp = timestamp
            self._lobs[timestamp] = lob

        return self._lobs[timestamp]

    def update(self, side, price_level, size, timestamp=0):
        """
        Apply a single update, creating a new LOB snapshot.

        Args:
            side: 'bid'/'b' or 'ask'/'a'
            price_level: Price level
            size: Quantity (0 to delete)
            timestamp: Timestamp for this snapshot

        Returns:
            The new LOB object
        """
        # Convert short form side to long form
        if side == 'b':
            side = 'bid'
        elif side == 'a':
            side = 'ask'
        return self.set_updates([(side, price_level, size)], timestamp)

    def __getitem__(self, timestamp_or_slice):
        """
        Return LOB at specific timestamp or slice.

        Args:
            timestamp_or_slice: Timestamp to retrieve, or slice object

        Returns:
            LOB object at timestamp, new LOBts for slice, or None if not found
        """
        if isinstance(timestamp_or_slice, slice):
            start = timestamp_or_slice.start
            stop = timestamp_or_slice.stop
            return self.get_range(start, stop)
        else:
            try:
                return self._lobs[timestamp_or_slice]
            except KeyError:
                return None

    def get_at_timestamp(self, timestamp):
        """
        Return LOB at specific timestamp.

        Args:
            timestamp: Timestamp to retrieve

        Returns:
            LOB object at timestamp, or None if not found
        """
        try:
            return self._lobs[timestamp]
        except KeyError:
            return None

    def get_range(self, start_ts, end_ts):
        """
        Return new LOBts containing only snapshots in time range [start_ts, end_ts].

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            New LOBts with filtered timestamps
        """
        result = LOBts(name=f"{self.name}_range", tick_size=self.tick_size, mode=self._mode)
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            result._lobs[ts] = self._lobs[ts]
        return result

    def __len__(self):
        """Return number of LOB snapshots stored."""
        return len(self._lobs)

    @property
    def len(self):
        """Return number of timestamps (for compatibility with example)."""
        return len(self._lobs)

    @property
    def len_ts(self):
        """Return duration: last timestamp - first timestamp."""
        timestamps = list(self._lobs.keys())
        if len(timestamps) < 2:
            return 0
        return timestamps[-1] - timestamps[0]

    def __contains__(self, timestamp):
        """Check if timestamp exists in the series."""
        return timestamp in self._lobs

    def __iter__(self):
        """Iterate over LOB objects in timestamp order."""
        return iter(self._lobs.values())

    def spread_ts(self, start_ts=None, end_ts=None):
        """
        Return spread time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and spread values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        spreads = []
        timestamps = []
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            spreads.append(self._lobs[ts].spread)
            timestamps.append(ts)

        return pd.Series(spreads, index=timestamps, name='spread')

    def midprice_ts(self, start_ts=None, end_ts=None):
        """
        Return mid-price time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and mid-price values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        midprices = []
        timestamps = []
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            latest = self._lobs[ts]
            midprices.append((latest.bid[0] + latest.ask[0]) / 2)
            timestamps.append(ts)

        return pd.Series(midprices, index=timestamps, name='midprice')

    def bid_ts(self, start_ts=None, end_ts=None):
        """
        Return best bid time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and bid price values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        bids = []
        timestamps = []
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            bids.append(self._lobs[ts].bid[0])
            timestamps.append(ts)

        return pd.Series(bids, index=timestamps, name='bid')

    def ask_ts(self, start_ts=None, end_ts=None):
        """
        Return best ask time series.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            pandas Series with timestamps as index and ask price values
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        asks = []
        timestamps = []
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            asks.append(self._lobs[ts].ask[0])
            timestamps.append(ts)

        return pd.Series(asks, index=timestamps, name='ask')

    @property
    def spread(self):
        """Return spread time series as property."""
        return self.spread_ts()

    @property
    def bid(self):
        """Return bid time series as property."""
        return self.bid_ts()

    @property
    def ask(self):
        """Return ask time series as property."""
        return self.ask_ts()

    @property
    def midprice(self):
        """Return mid-price time series as property."""
        return self.midprice_ts()

    @property
    def vw_midprice(self):
        """Return volume-weighted mid-price time series as property."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        vw_midprices = []
        timestamps = []
        for ts in self._lobs.keys():
            latest = self._lobs[ts]
            bid_price = latest.bid[0]
            bid_size = latest.bidq[0]
            ask_price = latest.ask[0]
            ask_size = latest.askq[0]
            if bid_size + ask_size == 0:
                import math
                vw_midprices.append(math.nan)
            else:
                vw_midprices.append((bid_price * bid_size + ask_price * ask_size) / (bid_size + ask_size))
            timestamps.append(ts)

        return pd.Series(vw_midprices, index=timestamps, name='vw_midprice')

    @property
    def vi(self):
        """Return volume imbalance time series as property."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for time series methods")

        vi_values = []
        timestamps = []
        for ts in self._lobs.keys():
            latest = self._lobs[ts]
            vi_values.append(latest.bidq[0] - latest.askq[0])
            timestamps.append(ts)

        return pd.Series(vi_values, index=timestamps, name='vi')

    @property
    def arrival_frequency(self):
        """
        Return total arrival frequency (quantity added to order book across all snapshots).

        In L2 order books, an arrival can be:
        - New level: a price level that didn't exist before
        - Quantity increase: existing level size increases (X -> Y where Y > X)

        Returns total quantity added (positive changes) across all transitions.
        """
        timestamps_list = list(self._lobs.keys())
        if len(timestamps_list) <= 1:
            return 0

        total_arrivals = 0
        for i in range(1, len(timestamps_list)):
            prev_ts = timestamps_list[i - 1]
            curr_ts = timestamps_list[i]
            prev_lob = self._lobs[prev_ts]
            curr_lob = self._lobs[curr_ts]

            arrivals = 0

            for price in curr_lob._bids.keys():
                new_qty = curr_lob._bids[price]
                if price not in prev_lob._bids:
                    arrivals += new_qty
                else:
                    old_qty = prev_lob._bids[price]
                    if new_qty > old_qty:
                        arrivals += (new_qty - old_qty)

            for price in curr_lob._asks.keys():
                new_qty = curr_lob._asks[price]
                if price not in prev_lob._asks:
                    arrivals += new_qty
                else:
                    old_qty = prev_lob._asks[price]
                    if new_qty > old_qty:
                        arrivals += (new_qty - old_qty)

            total_arrivals += arrivals

        return total_arrivals

    @property
    def cancel_frequency(self):
        """
        Return total cancel frequency (quantity removed from order book across all snapshots).

        In L2 order books, a cancel can be:
        - Full cancel: level completely removed (size goes to X -> 0 or level disappears)
        - Partial cancel: level size decreases (size goes from X -> Y where Y < X)

        Returns total quantity removed (negative changes) across all transitions.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for frequency methods")

        timestamps_list = list(self._lobs.keys())
        if len(timestamps_list) <= 1:
            return 0

        total_cancels = 0
        for i in range(1, len(timestamps_list)):
            prev_ts = timestamps_list[i - 1]
            curr_ts = timestamps_list[i]
            prev_lob = self._lobs[prev_ts]
            curr_lob = self._lobs[curr_ts]

            cancels = 0

            for price in prev_lob._bids.keys():
                old_qty = prev_lob._bids[price]
                if price not in curr_lob._bids:
                    cancels += old_qty
                else:
                    new_qty = curr_lob._bids[price]
                    if new_qty < old_qty:
                        cancels += (old_qty - new_qty)

            for price in prev_lob._asks.keys():
                old_qty = prev_lob._asks[price]
                if price not in curr_lob._asks:
                    cancels += old_qty
                else:
                    new_qty = curr_lob._asks[price]
                    if new_qty < old_qty:
                        cancels += (old_qty - new_qty)

            total_cancels += cancels

        return total_cancels

    def update_frequency(self):
        """Calculate update frequency (arrivals + cancels)."""
        return self.arrival_frequency() + self.cancel_frequency()

    def diff(self, other):
        """
        Calculate differences between this LOBts and another LOBts.

        Args:
            other: Another LOBts object to compare with

        Returns:
            List of (timestamp, bid_deltas, ask_deltas) tuples
        """
        results = []
        for ts in self._lobs.keys():
            if ts not in other._lobs:
                continue

            lob1 = self._lobs[ts]
            lob2 = other._lobs[ts]

            bid_deltas = []
            ask_deltas = []

            bids1 = dict(lob1._bids.items())
            bids2 = dict(lob2._bids.items())
            asks1 = dict(lob1._asks.items())
            asks2 = dict(lob2._asks.items())

            for price, qty in bids2.items():
                old_qty = bids1.get(price, 0.0)
                if qty != old_qty:
                    bid_deltas.append((price, qty))
            for price in bids1:
                if price not in bids2:
                    bid_deltas.append((price, 0.0))

            for price, qty in asks2.items():
                old_qty = asks1.get(price, 0.0)
                if qty != old_qty:
                    ask_deltas.append((price, qty))
            for price in asks1:
                if price not in asks2:
                    ask_deltas.append((price, 0.0))

            results.append((ts, bid_deltas, ask_deltas))

        return results

    def to_pd(self, start_ts=None, end_ts=None):
        """
        Export to pandas DataFrame.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            DataFrame with columns: timestamp, side, level, price, size
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for export methods")

        data = []
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self._lobs[ts]
            for level, (price, size) in enumerate(lob._bids.items()):
                data.append((ts, 'b', level, price, size))
            for level, (price, size) in enumerate(lob._asks.items()):
                data.append((ts, 'a', level, price, size))

        return pd.DataFrame(data, columns=['timestamp', 'side', 'level', 'price', 'size'])

    def to_np(self, start_ts=None, end_ts=None):
        """
        Export to numpy array.

        Args:
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)

        Returns:
            numpy array with shape (n, 5): [timestamp, side, level, price, size]
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for export methods")

        data = []
        for ts in self._lobs.keys():
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            lob = self._lobs[ts]
            for level, (price, size) in enumerate(lob._bids.items()):
                data.append([ts, 'b', level, price, size])
            for level, (price, size) in enumerate(lob._asks.items()):
                data.append([ts, 'a', level, price, size])

        return np.array(data, dtype=object)

    def to_csv(self, path, start_ts=None, end_ts=None):
        """
        Export to CSV file.

        Args:
            path: File path for CSV output
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)
        """
        df = self.to_pd(start_ts, end_ts)
        df.to_csv(path, index=False)

    def to_xlsx(self, path, start_ts=None, end_ts=None):
        """
        Export to XLSX file.

        Args:
            path: File path for XLSX output
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)
        """
        df = self.to_pd(start_ts, end_ts)
        df.to_excel(path, index=False, engine='openpyxl')

    def to_parquet(self, path, start_ts=None, end_ts=None):
        """
        Export to Parquet file.

        Args:
            path: File path for Parquet output
            start_ts: Start timestamp (inclusive)
            end_ts: End timestamp (inclusive)
        """
        df = self.to_pd(start_ts, end_ts)
        df.to_parquet(path, engine='pyarrow')

    def __repr__(self) -> str:
        return f"<LOBts[{self.name}] mode={self._mode} snapshots={len(self)}>"
