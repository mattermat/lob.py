"""
NASDAQ ITCH 5.0 parser.

Supports:
  - raw format   : MMDDYYYY.NASDAQ_ITCH50.gz  (2-byte length prefix per message)
  - SoupBinTCP   : S MMDDYY-v50.txt.gz        (2-byte length + 1-byte packet type)

Compression handled transparently for .gz and .zip files.
"""

import gzip
import struct
import zipfile
from dataclasses import dataclass
from typing import ClassVar, Generator, Optional

_PRICE_FACTOR = 10_000


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class AddOrder:
    """ITCH type A / F — new resting order added to the book."""
    type: str           # 'A' or 'F'
    timestamp: int      # nanoseconds since midnight
    order_ref: int
    side: str           # 'B' (buy) or 'S' (sell)
    shares: int
    symbol: str
    price: float
    mpid: str = ''      # only populated for type 'F'

    is_book_msg: ClassVar[bool] = True
    is_trade_msg: ClassVar[bool] = False


@dataclass
class DeleteOrder:
    """ITCH type D — full cancellation of a resting order."""
    type: str = 'D'
    timestamp: int = 0
    order_ref: int = 0
    symbol: str = ''
    side: str = ''
    price: float = 0.0
    shares: int = 0         # original order size to subtract from the price level

    is_book_msg: ClassVar[bool] = True
    is_trade_msg: ClassVar[bool] = False


@dataclass
class CancelOrder:
    """ITCH type X — partial cancellation of a resting order."""
    type: str = 'X'
    timestamp: int = 0
    order_ref: int = 0
    cancelled_shares: int = 0
    symbol: str = ''
    side: str = ''
    price: float = 0.0

    is_book_msg: ClassVar[bool] = True
    is_trade_msg: ClassVar[bool] = False


@dataclass
class ReplaceOrder:
    """ITCH type U — replace (cancel + re-add at new price/size)."""
    type: str = 'U'
    timestamp: int = 0
    order_ref: int = 0      # original reference
    new_order_ref: int = 0
    orig_shares: int = 0    # size of the cancelled order (subtract from old price level)
    shares: int = 0         # size of the new order (add to new price level)
    old_price: float = 0.0  # price of the cancelled order
    price: float = 0.0      # new price
    symbol: str = ''
    side: str = ''

    is_book_msg: ClassVar[bool] = True
    is_trade_msg: ClassVar[bool] = False


@dataclass
class ExecuteOrder:
    """ITCH type E / C — execution against a resting order (book + trade print)."""
    type: str = 'E'         # 'E' = at limit price, 'C' = at given execution price
    timestamp: int = 0
    order_ref: int = 0
    executed_shares: int = 0
    match_number: int = 0
    symbol: str = ''
    side: str = ''          # side of the resting order
    price: float = 0.0      # limit price (E) or execution price (C)

    is_book_msg: ClassVar[bool] = True
    is_trade_msg: ClassVar[bool] = True


@dataclass
class Trade:
    """ITCH type P / Q — trade print with no resting order reference."""
    type: str = 'P'         # 'P' = non-cross, 'Q' = cross
    timestamp: int = 0
    shares: int = 0
    symbol: str = ''
    price: float = 0.0
    match_number: int = 0
    side: str = ''          # aggressor side for P ('B'/'S'); '' for Q (cross)

    is_book_msg: ClassVar[bool] = False
    is_trade_msg: ClassVar[bool] = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ts(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 6], 'big')


def _parse_str(data: bytes, offset: int, length: int) -> str:
    return data[offset:offset + length].decode('ascii').rstrip()



def _open_file(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    if path.endswith('.zip'):
        zf = zipfile.ZipFile(path, 'r')
        return zf.open(zf.namelist()[0])
    return open(path, 'rb')


def _iter_raw(f) -> Generator[bytes, None, None]:
    """2-byte length prefix, then message bytes."""
    buf = f.read(2)
    while len(buf) == 2:
        (length,) = struct.unpack('>H', buf)
        msg = f.read(length)
        if len(msg) == length:
            yield msg
        buf = f.read(2)



# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class itch_parser:
    """
    NASDAQ ITCH 5.0 parser.

    Usage::

        parser = itch_parser("S101819-v50.txt.gz")
        for msg in parser.messages():
            if msg.symbol == "AAPL":
                ...
    """

    def __init__(self, path: str):
        self.path = path

    def messages(self) -> Generator:
        """
        Yield parsed ITCH messages.

        Yields AddOrder, DeleteOrder, CancelOrder, ReplaceOrder,
        ExecuteOrder, or Trade objects. System/directory messages are skipped.

        The generator maintains an internal order book (order_ref → symbol,
        side, price) so that all yielded messages carry a populated ``symbol``.
        Messages whose order_ref is not in the book (e.g. file starts mid-day)
        are silently skipped.
        """
        orders: dict[int, tuple[str, str, float, int]] = {}  # order_ref → (symbol, side, price, shares)
        f = _open_file(self.path)
        try:
            it = _iter_raw(f)
            for raw in it:
                msg = _parse(raw, orders)
                if msg is not None:
                    yield msg
        finally:
            f.close()


# ---------------------------------------------------------------------------
# Per-type parsers  (module-level to avoid repeated attribute lookup)
# ---------------------------------------------------------------------------

def _parse(data: bytes, orders: dict) -> Optional[object]:
    mtype = chr(data[0])
    if mtype in ('A', 'F'):
        return _add(data, mtype, orders)
    if mtype == 'D':
        return _delete(data, orders)
    if mtype == 'X':
        return _cancel(data, orders)
    if mtype == 'U':
        return _replace(data, orders)
    if mtype == 'E':
        return _execute(data, orders)
    if mtype == 'C':
        return _execute_price(data, orders)
    if mtype == 'P':
        return _trade(data)
    if mtype == 'Q':
        return _cross_trade(data)
    return None     # S, R, H, Y, L, V, W, K, J, N, … — skip


def _add(data: bytes, mtype: str, orders: dict) -> AddOrder:
    # type(1) locate(2) tracking(2) ts(6) order_ref(8) side(1) shares(4) stock(8) price(4)
    ts = _parse_ts(data, 5)
    (order_ref,) = struct.unpack_from('>Q', data, 11)
    side = chr(data[19])
    (shares,) = struct.unpack_from('>I', data, 20)
    symbol = _parse_str(data, 24, 8)
    (price_raw,) = struct.unpack_from('>I', data, 32)
    price = price_raw / _PRICE_FACTOR
    mpid = _parse_str(data, 36, 4) if mtype == 'F' else ''
    orders[order_ref] = (symbol, side, price, shares)
    return AddOrder(mtype, ts, order_ref, side, shares, symbol, price, mpid)


def _delete(data: bytes, orders: dict) -> Optional[DeleteOrder]:
    # type(1) locate(2) tracking(2) ts(6) order_ref(8)
    ts = _parse_ts(data, 5)
    (order_ref,) = struct.unpack_from('>Q', data, 11)
    info = orders.pop(order_ref, None)
    if info is None:
        return None
    symbol, side, price, shares = info
    return DeleteOrder('D', ts, order_ref, symbol, side, price, shares)


def _cancel(data: bytes, orders: dict) -> Optional[CancelOrder]:
    # type(1) locate(2) tracking(2) ts(6) order_ref(8) cancelled_shares(4)
    ts = _parse_ts(data, 5)
    (order_ref,) = struct.unpack_from('>Q', data, 11)
    (cancelled,) = struct.unpack_from('>I', data, 19)
    info = orders.get(order_ref)
    if info is None:
        return None
    symbol, side, price, orig_shares = info
    orders[order_ref] = (symbol, side, price, orig_shares - cancelled)
    return CancelOrder('X', ts, order_ref, cancelled, symbol, side, price)


def _replace(data: bytes, orders: dict) -> Optional[ReplaceOrder]:
    # type(1) locate(2) tracking(2) ts(6) orig_ref(8) new_ref(8) shares(4) price(4)
    ts = _parse_ts(data, 5)
    (orig_ref,) = struct.unpack_from('>Q', data, 11)
    (new_ref,) = struct.unpack_from('>Q', data, 19)
    (shares,) = struct.unpack_from('>I', data, 27)
    (price_raw,) = struct.unpack_from('>I', data, 31)
    price = price_raw / _PRICE_FACTOR
    info = orders.pop(orig_ref, None)
    if info is None:
        return None
    symbol, side, old_price, orig_shares = info
    orders[new_ref] = (symbol, side, price, shares)
    return ReplaceOrder('U', ts, orig_ref, new_ref, orig_shares, shares, old_price, price, symbol, side)


def _execute(data: bytes, orders: dict) -> Optional[ExecuteOrder]:
    # type(1) locate(2) tracking(2) ts(6) order_ref(8) exec_shares(4) match(8)
    ts = _parse_ts(data, 5)
    (order_ref,) = struct.unpack_from('>Q', data, 11)
    (exec_shares,) = struct.unpack_from('>I', data, 19)
    (match,) = struct.unpack_from('>Q', data, 23)
    info = orders.get(order_ref)
    if info is None:
        return None
    symbol, side, price, orig_shares = info
    remaining = orig_shares - exec_shares
    if remaining > 0:
        orders[order_ref] = (symbol, side, price, remaining)
    else:
        orders.pop(order_ref)
    return ExecuteOrder('E', ts, order_ref, exec_shares, match, symbol, side, price)


def _execute_price(data: bytes, orders: dict) -> Optional[ExecuteOrder]:
    # type(1) locate(2) tracking(2) ts(6) order_ref(8) exec_shares(4) match(8) printable(1) exec_price(4)
    ts = _parse_ts(data, 5)
    (order_ref,) = struct.unpack_from('>Q', data, 11)
    (exec_shares,) = struct.unpack_from('>I', data, 19)
    (match,) = struct.unpack_from('>Q', data, 23)
    (price_raw,) = struct.unpack_from('>I', data, 32)
    price = price_raw / _PRICE_FACTOR
    info = orders.get(order_ref)
    if info is None:
        return None
    symbol, side, limit_price, orig_shares = info
    remaining = orig_shares - exec_shares
    if remaining > 0:
        orders[order_ref] = (symbol, side, limit_price, remaining)
    else:
        orders.pop(order_ref)
    return ExecuteOrder('C', ts, order_ref, exec_shares, match, symbol, side, price)


def _trade(data: bytes) -> Trade:
    # type(1) locate(2) tracking(2) ts(6) order_ref(8) side(1) shares(4) stock(8) price(4) match(8)
    ts = _parse_ts(data, 5)
    side = chr(data[19])
    (shares,) = struct.unpack_from('>I', data, 20)
    symbol = _parse_str(data, 24, 8)
    (price_raw,) = struct.unpack_from('>I', data, 32)
    price = price_raw / _PRICE_FACTOR
    (match,) = struct.unpack_from('>Q', data, 36)
    return Trade('P', ts, shares, symbol, price, match, side)


def _cross_trade(data: bytes) -> Trade:
    # type(1) locate(2) tracking(2) ts(6) shares(8) stock(8) cross_price(4) match(8) cross_type(1)
    ts = _parse_ts(data, 5)
    (shares,) = struct.unpack_from('>Q', data, 11)
    symbol = _parse_str(data, 19, 8)
    (price_raw,) = struct.unpack_from('>I', data, 27)
    price = price_raw / _PRICE_FACTOR
    (match,) = struct.unpack_from('>Q', data, 31)
    return Trade('Q', ts, shares, symbol, price, match, '')
