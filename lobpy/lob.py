import time

from .sorteddict import SortedDict

def neg(x: int) -> int:
    return -x

class LoB():

    def __init__(self, name=None, *, bids=[], asks=[]) -> None:
        if name is None:
            name = f"lob{id(self)}"
        self.name = name
        self._bids = SortedDict(neg)
        self._asks = SortedDict()
        self.tick_size = 1
        self.timestamp = int(time.time()*1000)

    def set_tick_size(self, tick_size) -> None:
        self.tick_size = tick_size

    def set_snapshot(self, bids, asks, timestamp=0):
        """
        align the order book to a snapshot
        """
        for b in bids:
            self._bids[b[0]] = b[1]
        for a in asks:
            self._asks[a[0]] = a[1]
        self.timestamp = timestamp

    def delete_level(self, side, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp

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

    def delete_ask_level(self, price_level, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        try:
            del self._asks[price_level]
        except KeyError:
            pass # TODO: error message
        return
    
    def delete_bid_level(self, price_level, timestamp=0):
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
            self.delete_level(side, price_level, timestamp)
            return

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

    def update_ask(self, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if size == 0:
            self.delete_ask_level(price_level)
        else:
            self._asks[price_level] = size
        return

    def update_bid(self, price_level, size, timestamp=0):
        if timestamp != 0:
            self.timestamp = timestamp
        if size == 0:
            self.delete_bid_level(price_level)
        else:
            self._bids[price_level] = size
        return

    @property
    def ask(self):
        """
        get a tuple containing the best ask price and size
        """
        try:
            return self._asks.peekitem(0)
        except IndexError:
            return (.0, .0)

    @property
    def bid(self):
        """
        get a tuple containing the best bid price and size
        """
        try:
            return self._bids.peekitem(0)

        except IndexError:
            return (.0, .0)

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

    def __repr__(self) -> str:
        return f"<Book[{self.name}]>"