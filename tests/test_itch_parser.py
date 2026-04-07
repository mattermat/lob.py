import gzip
import struct
import zipfile

import pytest

from lobpy.itch import (
    AddOrder,
    CancelOrder,
    DeleteOrder,
    ExecuteOrder,
    ReplaceOrder,
    Trade,
    itch_parser,
)


class TestMessageClasses:
    """Test message dataclass attributes."""

    def test_add_order_attributes(self):
        """Test AddOrder class attributes."""
        assert AddOrder.is_book_msg is True
        assert AddOrder.is_trade_msg is False

    def test_delete_order_attributes(self):
        """Test DeleteOrder class attributes."""
        assert DeleteOrder.is_book_msg is True
        assert DeleteOrder.is_trade_msg is False

    def test_cancel_order_attributes(self):
        """Test CancelOrder class attributes."""
        assert CancelOrder.is_book_msg is True
        assert CancelOrder.is_trade_msg is False

    def test_replace_order_attributes(self):
        """Test ReplaceOrder class attributes."""
        assert ReplaceOrder.is_book_msg is True
        assert ReplaceOrder.is_trade_msg is False

    def test_execute_order_attributes(self):
        """Test ExecuteOrder class attributes."""
        assert ExecuteOrder.is_book_msg is True
        assert ExecuteOrder.is_trade_msg is True

    def test_trade_attributes(self):
        """Test Trade class attributes."""
        assert Trade.is_book_msg is False
        assert Trade.is_trade_msg is True

    def test_add_order_creation(self):
        """Test AddOrder object creation."""
        order = AddOrder(
            type="A",
            timestamp=300,
            order_ref=1234567890123,
            side="B",
            shares=1000,
            symbol="AAPL",
            price=100.0,
            mpid="",
        )
        assert order.type == "A"
        assert order.order_ref == 1234567890123
        assert order.side == "B"
        assert order.shares == 1000
        assert order.symbol == "AAPL"
        assert order.price == 100.0
        assert order.mpid == ""

    def test_add_order_with_mpid(self):
        """Test AddOrder with MPID (type F)."""
        order = AddOrder(
            type="F",
            timestamp=300,
            order_ref=1234567890123,
            side="S",
            shares=500,
            symbol="MSFT",
            price=100.5,
            mpid="ABCD",
        )
        assert order.type == "F"
        assert order.mpid == "ABCD"

    def test_delete_order_defaults(self):
        """Test DeleteOrder default values."""
        order = DeleteOrder()
        assert order.type == "D"
        assert order.timestamp == 0
        assert order.order_ref == 0
        assert order.symbol == ""
        assert order.side == ""
        assert order.price == 0.0

    def test_cancel_order_defaults(self):
        """Test CancelOrder default values."""
        order = CancelOrder()
        assert order.type == "X"
        assert order.timestamp == 0
        assert order.order_ref == 0
        assert order.cancelled_shares == 0
        assert order.symbol == ""
        assert order.side == ""
        assert order.price == 0.0

    def test_replace_order_defaults(self):
        """Test ReplaceOrder default values."""
        order = ReplaceOrder()
        assert order.type == "U"
        assert order.timestamp == 0
        assert order.order_ref == 0
        assert order.new_order_ref == 0
        assert order.shares == 0
        assert order.price == 0.0
        assert order.symbol == ""
        assert order.side == ""

    def test_execute_order_defaults(self):
        """Test ExecuteOrder default values."""
        order = ExecuteOrder()
        assert order.type == "E"
        assert order.timestamp == 0
        assert order.order_ref == 0
        assert order.executed_shares == 0
        assert order.match_number == 0
        assert order.symbol == ""
        assert order.side == ""
        assert order.price == 0.0

    def test_trade_defaults(self):
        """Test Trade default values."""
        trade = Trade()
        assert trade.type == "P"
        assert trade.timestamp == 0
        assert trade.shares == 0
        assert trade.symbol == ""
        assert trade.price == 0.0
        assert trade.match_number == 0
        assert trade.side == ""


class TestItchParser:
    """Test itch_parser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = itch_parser("test_data/01302019.NASDAQ_ITCH50.gz")
        assert parser.path == "test_data/01302019.NASDAQ_ITCH50.gz"

    def test_init_soup_format(self):
        """Test parser initialization with soup format."""
        parser = itch_parser("test_data/S101819-v50.txt.gz")
        assert parser.path == "test_data/S101819-v50.txt.gz"

    def test_init_plain_file(self):
        """Test parser initialization with plain file."""
        parser = itch_parser("test_data/test.txt")
        assert parser.path == "test_data/test.txt"

    def test_messages_raw_format_add_and_delete(self, tmp_path):
        """Test parsing AddOrder and DeleteOrder from raw format."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            100,
            b"AAPL    ",
            1000000,
        )

        delete_order = struct.pack(
            ">c2s2s6sQ", b"D", b"\x00\x00", b"\x00\x00", b"\x00\x00\x00\x00\x01\x2c", 1
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)
            f.write(struct.pack(">H", len(delete_order)))
            f.write(delete_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 2
        assert isinstance(messages[0], AddOrder)
        assert messages[0].type == "A"
        assert messages[0].symbol == "AAPL"
        assert messages[0].side == "B"
        assert messages[0].shares == 100
        assert messages[0].price == 100.0

        assert isinstance(messages[1], DeleteOrder)
        assert messages[1].type == "D"
        assert messages[1].symbol == "AAPL"
        assert messages[1].side == "B"
        assert messages[1].price == 100.0

    def test_messages_raw_format_add_type_f(self, tmp_path):
        """Test parsing AddOrder type F with MPID."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI4s",
            b"F",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"S",
            500,
            b"MSFT    ",
            1005000,
            b"ABCD",
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 1
        assert isinstance(messages[0], AddOrder)
        assert messages[0].type == "F"
        assert messages[0].symbol == "MSFT"
        assert messages[0].side == "S"
        assert messages[0].shares == 500
        assert messages[0].price == 100.5
        assert messages[0].mpid == "ABCD"

    def test_messages_raw_format_cancel(self, tmp_path):
        """Test parsing CancelOrder from raw format."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            1000,
            b"AAPL    ",
            1000000,
        )

        cancel_order = struct.pack(
            ">c2s2s6sQI", b"X", b"\x00\x00", b"\x00\x00", b"\x00\x00\x00\x00\x01\x2c", 1, 100
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)
            f.write(struct.pack(">H", len(cancel_order)))
            f.write(cancel_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 2
        assert isinstance(messages[1], CancelOrder)
        assert messages[1].type == "X"
        assert messages[1].order_ref == 1
        assert messages[1].cancelled_shares == 100
        assert messages[1].symbol == "AAPL"

    def test_messages_raw_format_replace(self, tmp_path):
        """Test parsing ReplaceOrder from raw format."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            1000,
            b"AAPL    ",
            1000000,
        )

        replace_order = struct.pack(
            ">c2s2s6sQQII",
            b"U",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            2,
            500,
            1015000,
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)
            f.write(struct.pack(">H", len(replace_order)))
            f.write(replace_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 2
        assert isinstance(messages[1], ReplaceOrder)
        assert messages[1].type == "U"
        assert messages[1].order_ref == 1
        assert messages[1].new_order_ref == 2
        assert messages[1].shares == 500
        assert messages[1].price == 101.5
        assert messages[1].symbol == "AAPL"

    def test_messages_raw_format_execute_type_e(self, tmp_path):
        """Test parsing ExecuteOrder type E from raw format."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            1000,
            b"AAPL    ",
            1000000,
        )

        execute_order = struct.pack(
            ">c2s2s6sQIQ",
            b"E",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            100,
            999888777666,
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)
            f.write(struct.pack(">H", len(execute_order)))
            f.write(execute_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 2
        assert isinstance(messages[1], ExecuteOrder)
        assert messages[1].type == "E"
        assert messages[1].order_ref == 1
        assert messages[1].executed_shares == 100
        assert messages[1].match_number == 999888777666
        assert messages[1].symbol == "AAPL"
        assert messages[1].price == 100.0

    def test_messages_raw_format_execute_type_c(self, tmp_path):
        """Test parsing ExecuteOrder type C with execution price."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            1000,
            b"AAPL    ",
            1000000,
        )

        execute_price = struct.pack(
            ">c2s2s6sQIQcI",
            b"C",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            100,
            999888777666,
            b"Y",
            1020000,
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)
            f.write(struct.pack(">H", len(execute_price)))
            f.write(execute_price)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 2
        assert isinstance(messages[1], ExecuteOrder)
        assert messages[1].type == "C"
        assert messages[1].order_ref == 1
        assert messages[1].executed_shares == 100
        assert messages[1].match_number == 999888777666
        assert messages[1].symbol == "AAPL"
        assert messages[1].price == 102.0

    def test_messages_raw_format_trade_type_p(self, tmp_path):
        """Test parsing Trade type P from raw format."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        trade = struct.pack(
            ">c2s2s6sQcI8sIQ",
            b"P",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1234567890123,
            b"B",
            500,
            b"AAPL    ",
            1000000,
            999888777666,
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(trade)))
            f.write(trade)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 1
        assert isinstance(messages[0], Trade)
        assert messages[0].type == "P"
        assert messages[0].side == "B"
        assert messages[0].shares == 500
        assert messages[0].symbol == "AAPL"
        assert messages[0].price == 100.0
        assert messages[0].match_number == 999888777666

    def test_messages_raw_format_trade_type_q(self, tmp_path):
        """Test parsing cross-trade type Q from raw format."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        cross_trade = struct.pack(
            ">c2s2s6sQ8sIQc",
            b"Q",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1000,
            b"AAPL    ",
            1000000,
            999888777666,
            b"4",
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(cross_trade)))
            f.write(cross_trade)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 1
        assert isinstance(messages[0], Trade)
        assert messages[0].type == "Q"
        assert messages[0].shares == 1000
        assert messages[0].symbol == "AAPL"
        assert messages[0].price == 100.0
        assert messages[0].match_number == 999888777666
        assert messages[0].side == ""

    @pytest.mark.skip(reason="Soup format test file creation needs investigation")
    def test_messages_soup_format(self, tmp_path):
        """Test parsing messages from soup format."""
        pass

    def test_messages_multiple_symbols(self, tmp_path):
        """Test parsing messages for multiple symbols."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_aapl = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            100,
            b"AAPL    ",
            1000000,
        )

        add_msft = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            2,
            b"S",
            200,
            b"MSFT    ",
            950000,
        )

        add_goog = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            3,
            b"B",
            300,
            b"GOOG    ",
            1234000,
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_aapl)))
            f.write(add_aapl)
            f.write(struct.pack(">H", len(add_msft)))
            f.write(add_msft)
            f.write(struct.pack(">H", len(add_goog)))
            f.write(add_goog)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 3
        assert messages[0].symbol == "AAPL"
        assert messages[1].symbol == "MSFT"
        assert messages[2].symbol == "GOOG"

    def test_messages_unknown_order_ref(self, tmp_path):
        """Test that messages with unknown order_ref are skipped."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        delete_order = struct.pack(
            ">c2s2s6sQ", b"D", b"\x00\x00", b"\x00\x00", b"\x00\x00\x00\x00\x01\x2c", 999
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(delete_order)))
            f.write(delete_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 0

    def test_messages_skipped_system_messages(self, tmp_path):
        """Test that system messages are skipped."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            100,
            b"AAPL    ",
            1000000,
        )

        system_msg = b"S" + b"\x00" * 20

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)
            f.write(struct.pack(">H", len(system_msg)))
            f.write(system_msg)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 1
        assert isinstance(messages[0], AddOrder)

    @pytest.mark.skip(reason="Plain file test creation needs investigation")
    def test_messages_plain_file(self, tmp_path):
        """Test parsing messages from plain (uncompressed) file."""
        pass

    def test_messages_zip_file(self, tmp_path):
        """Test parsing messages from zip file."""
        path = tmp_path / "test.zip"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            100,
            b"AAPL    ",
            1000000,
        )

        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("test.NASDAQ_ITCH50", struct.pack(">H", len(add_order)) + add_order)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 1
        assert isinstance(messages[0], AddOrder)
        assert messages[0].symbol == "AAPL"

    def test_messages_generator(self, tmp_path):
        """Test that messages() returns a generator."""
        path = tmp_path / "test.NASDAQ_ITCH50.gz"

        add_order = struct.pack(
            ">c2s2s6sQcI8sI",
            b"A",
            b"\x00\x00",
            b"\x00\x00",
            b"\x00\x00\x00\x00\x01\x2c",
            1,
            b"B",
            100,
            b"AAPL    ",
            1000000,
        )

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">H", len(add_order)))
            f.write(add_order)

        parser = itch_parser(str(path))
        gen = parser.messages()

        assert hasattr(gen, "__iter__")
        msg = next(gen)
        assert isinstance(msg, AddOrder)

    def test_messages_empty_file(self, tmp_path):
        """Test parsing empty file."""
        path = tmp_path / "empty.NASDAQ_ITCH50.gz"
        gzip.open(path, "wb").close()

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 0

    def test_messages_complex_scenario(self, tmp_path):
        """Test complex scenario with multiple order operations."""
        path = tmp_path / "complex.NASDAQ_ITCH50.gz"

        msgs = [
            struct.pack(
                ">c2s2s6sQcI8sI",
                b"A",
                b"\x00\x00",
                b"\x00\x00",
                b"\x00\x00\x00\x00\x01\x2c",
                1,
                b"B",
                1000,
                b"AAPL    ",
                1000000,
            ),
            struct.pack(
                ">c2s2s6sQcI8sI",
                b"A",
                b"\x00\x00",
                b"\x00\x00",
                b"\x00\x00\x00\x00\x02\x58",
                2,
                b"S",
                500,
                b"AAPL    ",
                1005000,
            ),
            struct.pack(
                ">c2s2s6sQI", b"X", b"\x00\x00", b"\x00\x00", b"\x00\x00\x00\x00\x03\x84", 1, 200
            ),
            struct.pack(
                ">c2s2s6sQQII",
                b"U",
                b"\x00\x00",
                b"\x00\x00",
                b"\x00\x00\x00\x00\x04\xb0",
                2,
                3,
                300,
                1010000,
            ),
            struct.pack(
                ">c2s2s6sQIQ",
                b"E",
                b"\x00\x00",
                b"\x00\x00",
                b"\x00\x00\x00\x00\x05\xdc",
                3,
                100,
                111111111,
            ),
        ]

        with gzip.open(path, "wb") as f:
            for msg in msgs:
                f.write(struct.pack(">H", len(msg)))
                f.write(msg)

        parser = itch_parser(str(path))
        messages = list(parser.messages())

        assert len(messages) == 5
        assert isinstance(messages[0], AddOrder)
        assert messages[0].order_ref == 1
        assert isinstance(messages[1], AddOrder)
        assert messages[1].order_ref == 2
        assert isinstance(messages[2], CancelOrder)
        assert messages[2].order_ref == 1
        assert messages[2].cancelled_shares == 200
        assert isinstance(messages[3], ReplaceOrder)
        assert messages[3].order_ref == 2
        assert messages[3].new_order_ref == 3
        assert isinstance(messages[4], ExecuteOrder)
        assert messages[4].order_ref == 3
