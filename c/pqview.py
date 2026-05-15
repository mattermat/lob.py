#!/usr/bin/env python3
"""
pqview — Fast terminal Parquet/CSV table viewer.

Usage:
    python pqview.py <file.parquet|file.csv>

Keys:
    arrows / hjkl      scroll
    PgUp/PgDn / Space  page scroll
    g / G              go to first / last row
    0 / $              go to first / last column
    /                  search in selected column
    n / N              next / previous match
    s / S              cycle selected column
    c                  jump to column (prompt)
    Ctrl+K             command mode
    r                  reload file
    ?                  toggle help overlay
    q                  quit

Commands (Ctrl+K):
    goto end           go to last row
    goto top           go to first row
"""

from __future__ import annotations

import curses
import os
import re
import sys
import time
from typing import Any

# ── Minimum Python version ────────────────────────────────────────────────
MIN_PY = (3, 9)
if sys.version_info < MIN_PY:
    sys.exit(f"Python {'.'.join(map(str, MIN_PY))}+ required")

# ── Data layer ────────────────────────────────────────────────────────────

class ParquetData:
    """Efficient parquet/CSV loader with row-cache and lazy formatting."""

    CHUNK = 50  # rows per cache bucket

    def __init__(self, path: str) -> None:
        self.path = path
        self.df: Any = None
        self.columns: list[str] = []
        self.num_rows = 0
        self.num_cols = 0
        self._dtypes: list[str] = []
        self._col_widths: list[int] = []
        self._load_time = 0.0
        self._row_cache: dict[int, list[list[str]]] = {}
        self._load()

    # ── file I/O ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        import pandas as pd

        t0 = time.perf_counter()
        p = self.path

        if p.lower().endswith((".parquet", ".pq")):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(p)
            self.df = pf.read().to_pandas()
        else:
            self.df = pd.read_csv(p)

        self.num_rows = len(self.df)
        self.columns = list(self.df.columns)
        self.num_cols = len(self.columns)
        self._dtypes = [str(self.df.iloc[:, i].dtype) for i in range(self.num_cols)]
        self._load_time = time.perf_counter() - t0

        self._compute_column_widths()
        self._row_cache.clear()

    def reload(self) -> None:
        self._load()

    # ── column width estimation ───────────────────────────────────────────

    def _compute_column_widths(self) -> None:
        self._col_widths = []
        sample_n = min(100, self.num_rows)

        for i, col in enumerate(self.columns):
            dtype = self._dtypes[i]
            hw = max(len(col), len(dtype)) + 3  # " name [dtype]"

            max_vw = 0
            if sample_n > 0:
                col_vals = self.df.iloc[:sample_n, i]
                for v in col_vals:
                    s = self._fmt_val(v, dtype)
                    if s:
                        max_vw = max(max_vw, len(s))

            w = max(hw, min(max_vw + 2, 32))
            self._col_widths.append(w + 1)  # +1 for leading space

    # ── cache management ──────────────────────────────────────────────────

    def _ensure_chunk(self, chunk_start: int) -> None:
        if chunk_start in self._row_cache:
            return
        chunk_end = min(chunk_start + self.CHUNK, self.num_rows)
        rows: list[list[str]] = []
        for r in range(chunk_start, chunk_end):
            row = [self._fmt_val(self.df.iloc[r, c], self._dtypes[c])
                   for c in range(self.num_cols)]
            rows.append(row)
        self._row_cache[chunk_start] = rows

    def get_cell(self, row: int, col: int) -> str:
        cs = (row // self.CHUNK) * self.CHUNK
        self._ensure_chunk(cs)
        return self._row_cache[cs][row - cs][col]

    def get_row_batch(self, start: int, count: int) -> list[list[str]]:
        """Fetch `count` formatted rows starting at `start`."""
        end = min(start + count, self.num_rows)
        # Pre-cache all needed chunks
        needed = {(r // self.CHUNK) * self.CHUNK for r in range(start, end)}
        for cs in needed:
            self._ensure_chunk(cs)

        result: list[list[str]] = []
        for r in range(start, end):
            cs = (r // self.CHUNK) * self.CHUNK
            result.append(self._row_cache[cs][r - cs])
        return result

    # ── search ────────────────────────────────────────────────────────────

    def search_all(self, col: int, pattern: str) -> list[int]:
        """Return all row indices where `pattern` matches column `col`."""
        if not pattern:
            return []
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        matches: list[int] = []
        col_vals = self.df.iloc[:, col]
        for r in range(self.num_rows):
            if regex.search(str(col_vals.iloc[r])):
                matches.append(r)
        return matches

    # ── display helper ────────────────────────────────────────────────────

    @staticmethod
    def _fmt_val(v: Any, dtype: str) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if v != v:  # NaN
                return ""
            av = abs(v)
            if av == 0:
                return "0"
            if av >= 1e8 or (0 < av < 1e-5):
                return f"{v:.6g}"
            # Check if it's actually an integer in float clothing (e.g. timestamp)
            if v == int(v) and abs(v) < 2**53:
                # it's an int-typed-column stored as float
                if "int" in dtype:
                    return str(int(v))
            if av >= 1000:
                return f"{v:.2f}"
            return f"{v:.6g}"
        return str(v)

    def col_width(self, col: int) -> int:
        return self._col_widths[col]


# ── Terminal UI ───────────────────────────────────────────────────────────

class PqView:
    """Curses-based table viewer for Parquet/CSV files."""

    def __init__(self, stdscr: curses.window, path: str) -> None:
        self.scr = stdscr
        self.path = path

        # Colors
        self._init_colors()

        # Load
        self._status_msg("Loading...")
        self._render_splash()
        self.data = ParquetData(path)

        # Viewport
        self.row = 0
        self.col = 0
        self.sel_col = 0

        # Search state
        self.search_mode = False
        self.search_buf = ""
        self.search_results: list[int] = []
        self.search_idx = -1

        # Command state
        self.command_mode = False
        self.command_buf = ""

        # UI state
        self.show_help = False
        self.status_text = ""
        self.status_until = 0.0  # perf_counter

        # Run
        self._run()

    # ── colors ───────────────────────────────────────────────────────────

    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        # indexed from 1
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)    # header
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)    # sel col
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # search hit
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE)   # top/bottom bar
        curses.init_pair(5, curses.COLOR_CYAN, -1)                    # row numbers
        curses.init_pair(6, curses.COLOR_BLACK, -1)                   # empty cells

    # ── main loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        curses.curs_set(0)
        self.scr.timeout(30)  # ~33 fps
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

        while True:
            self._render()
            key = self.scr.getch()
            if key == curses.KEY_RESIZE:
                continue
            if key == -1:
                continue
            if not self._handle_key(key):
                break

    # ── key dispatch ─────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        """Return False to exit."""
        if self.search_mode:
            return self._search_key(key)
        if self.command_mode:
            return self._command_key(key)

        mr = max(self.data.num_rows - 1, 0)
        mc = max(self.data.num_cols - 1, 0)
        pg = max(1, self._body_height() - 1)

        if key == ord("q"):
            return False

        # Mouse
        if key == curses.KEY_MOUSE:
            self._handle_mouse()
            return True

        # Movement
        elif key in (ord("h"), curses.KEY_LEFT):
            self.col = max(0, self.col - 1)
        elif key in (ord("l"), curses.KEY_RIGHT):
            self.col = min(mc, self.col + 1)
        elif key in (ord("j"), curses.KEY_DOWN):
            self.row = min(mr, self.row + 1)
        elif key in (ord("k"), curses.KEY_UP):
            self.row = max(0, self.row - 1)
        elif key == ord("g"):
            self.row = 0
        elif key == ord("G"):
            self.row = mr
        elif key == ord("0"):
            self.col = 0
        elif key == ord("$"):
            self.col = mc
        elif key in (curses.KEY_NPAGE, ord(" ")):
            self.row = min(mr, self.row + pg)
        elif key in (curses.KEY_PPAGE, curses.KEY_BTAB):
            self.row = max(0, self.row - pg)
        elif key == curses.KEY_HOME:
            self.row = 0
        elif key == curses.KEY_END:
            self.row = mr

        # Selection
        elif key == ord("s"):
            self.sel_col = (self.sel_col + 1) % (mc + 1)
        elif key == ord("S"):
            self.sel_col = (self.sel_col - 1) % (mc + 1)

        # Search
        elif key == ord("/"):
            self._enter_search()
        elif key == ord("n"):
            self._search_step(1)
        elif key == ord("N"):
            self._search_step(-1)

        # Command mode
        elif key == 11:  # Ctrl+K
            self._enter_command()

        # Actions
        elif key == ord("c"):
            self._prompt_jump_column()
        elif key == ord("r"):
            self._status_msg("Reloading...")
            self.data.reload()
            self.row = self.col = 0
        elif key == ord("?"):
            self.show_help = not self.show_help

        return True

    # ── mouse ────────────────────────────────────────────────────────────

    def _handle_mouse(self) -> None:
        try:
            _id, x, y, _z, bstate = curses.getmouse()
        except curses.error:
            return

        scroll_lines = 3
        mr = max(self.data.num_rows - 1, 0)

        if bstate & curses.BUTTON4_PRESSED:  # scroll up
            self.row = max(0, self.row - scroll_lines)
        elif bstate & curses.BUTTON5_PRESSED:  # scroll down
            self.row = min(mr, self.row + scroll_lines)

    # ── search ───────────────────────────────────────────────────────────

    def _enter_search(self) -> None:
        self.search_mode = True
        self.search_buf = ""
        self.search_results.clear()
        self.search_idx = -1
        self._status_msg("Search: ")

    def _search_key(self, key: int) -> bool:
        if key == 27:  # ESC
            self.search_mode = False
            return True
        if key in (10, 13):  # Enter
            self.search_mode = False
            return True
        if key in (curses.KEY_BACKSPACE, 127, 8):
            if self.search_buf:
                self.search_buf = self.search_buf[:-1]
                self._exec_search()
            return True
        if 32 <= key <= 126:
            self.search_buf += chr(key)
            self._exec_search()
            return True
        return True

    def _exec_search(self) -> None:
        self.search_results = self.data.search_all(self.sel_col, self.search_buf)
        self.search_idx = 0
        if self.search_results:
            self.row = self.search_results[0]

    def _search_step(self, direction: int) -> None:
        if not self.search_results:
            return
        self.search_idx = (self.search_idx + direction) % len(self.search_results)
        self.row = self.search_results[self.search_idx]

    # ── command mode ──────────────────────────────────────────────────────

    def _enter_command(self) -> None:
        self.command_mode = True
        self.command_buf = ""
        self._status_msg(":")

    def _command_key(self, key: int) -> bool:
        if key == 27:  # ESC
            self.command_mode = False
            self._status_msg("")
            return True
        if key in (10, 13):  # Enter
            self.command_mode = False
            self._exec_command(self.command_buf.strip())
            return True
        if key in (curses.KEY_BACKSPACE, 127, 8):
            if self.command_buf:
                self.command_buf = self.command_buf[:-1]
            return True
        if 32 <= key <= 126:
            self.command_buf += chr(key)
            return True
        return True

    def _exec_command(self, cmd: str) -> None:
        if not cmd:
            return
        parts = cmd.split()
        if not parts:
            return

        command = parts[0].lower()

        if command == "goto":
            if len(parts) >= 2:
                target = parts[1].lower()
                mr = max(self.data.num_rows - 1, 0)
                if target in ("end", "bottom"):
                    self.row = mr
                    self._status_msg(f"Goto end (row {self.row:,})")
                elif target in ("top", "start", "begin", "beginning"):
                    self.row = 0
                    self._status_msg(f"Goto top (row 0)")
                else:
                    self._status_msg(f"Unknown goto target: {target}")
            else:
                self._status_msg("Usage: goto <top|end>")
        else:
            self._status_msg(f"Unknown command: {cmd}")

    # ── column jump ──────────────────────────────────────────────────────

    def _prompt_jump_column(self) -> None:
        """Inline bottom-bar prompt for jumping to a column."""
        curses.curs_set(1)
        curses.echo()
        buf = ""

        try:
            while True:
                self._render()
                self._draw_status_raw(f" Jump to column: {buf}_")
                self.scr.refresh()
                ch = self.scr.getch()
                if ch == 27:
                    break
                if ch in (10, 13):
                    self._jump_col(buf.strip())
                    break
                if ch in (curses.KEY_BACKSPACE, 127, 8):
                    buf = buf[:-1]
                elif 32 <= ch <= 126:
                    buf += chr(ch)
                else:
                    pass  # ignore other keys during prompt
        finally:
            curses.noecho()
            curses.curs_set(0)

    def _jump_col(self, s: str) -> None:
        if not s:
            return
        # Numeric
        try:
            idx = int(s)
            if 0 <= idx < self.data.num_cols:
                self.col = idx
                return
        except ValueError:
            pass
        # Name prefix / substring
        sl = s.lower()
        for i, name in enumerate(self.data.columns):
            if name.lower().startswith(sl):
                self.col = i
                return
        for i, name in enumerate(self.data.columns):
            if sl in name.lower():
                self.col = i
                return

    # ── layout helpers ────────────────────────────────────────────────────

    def _body_height(self) -> int:
        my = self.scr.getmaxyx()[0]
        if my <= 0:
            return 1
        return max(1, my - 3)  # top bar + header + status

    def _rn_width(self) -> int:
        return max(6, len(str(self.data.num_rows)) + 2)

    def _visible_cols(self) -> tuple[list[int], list[int]]:
        """Return (indices, x-positions) for visible columns."""
        mx = self.scr.getmaxyx()[1]
        if mx <= 0:
            return [], []
        avail = max(20, mx - self._rn_width() - 2)
        indices: list[int] = []
        pos: list[int] = []
        x = 0
        for c in range(self.col, self.data.num_cols):
            w = self.data.col_width(c)
            if x + w > avail:
                break
            indices.append(c)
            pos.append(x)
            x += w
        return indices, pos

    # ── status messages ──────────────────────────────────────────────────

    def _status_msg(self, msg: str, duration: float = 3.0) -> None:
        self.status_text = msg
        self.status_until = time.perf_counter() + duration

    def _status_line(self) -> str:
        """Build the status line."""
        parts = [f"row {self.row:,}/{self.data.num_rows:,}"]
        if 0 <= self.sel_col < self.data.num_cols:
            parts.append(f"col {self.sel_col}:{self.data.columns[self.sel_col]}")
        if self.search_results:
            parts.append(f"match {self.search_idx + 1}/{len(self.search_results)}")
        txt = " | ".join(parts)

        # Override with temporary status message if fresh
        if time.perf_counter() < self.status_until and self.status_text:
            txt = self.status_text + "  " + txt
        if self.search_mode:
            txt = f"/{self.search_buf}  " + txt
        if self.command_mode:
            txt = f":{self.command_buf}  " + txt
        return txt

    # ── safe drawing helper ──────────────────────────────────────────────

    @staticmethod
    def _sadd(win: curses.window, y: int, x: int, s: str, attr: int = 0) -> None:
        """Safe addstr — silently ignores boundary errors."""
        try:
            win.addstr(y, x, s, attr)
        except curses.error:
            pass

    # ── render ────────────────────────────────────────────────────────────

    def _render(self) -> None:
        try:
            self.scr.erase()
        except curses.error:
            return
        my, mx = self.scr.getmaxyx()
        if my < 4 or mx < 10:
            self._sadd(self.scr, 0, 0, "Terminal too small")
            self.scr.refresh()
            return
        self._draw_top_bar(mx)
        self._draw_table(my, mx)
        self._draw_status_raw(self._status_line())
        if self.show_help:
            self._draw_help(my, mx)
        try:
            self.scr.refresh()
        except curses.error:
            pass

    def _render_splash(self) -> None:
        """One-frame splash while loading."""
        self.scr.erase()
        my, mx = self.scr.getmaxyx()
        msg = f"Loading {os.path.basename(self.path)}..."
        self._sadd(self.scr, my // 2, max(0, (mx - len(msg)) // 2), msg)
        self.scr.refresh()

    def _draw_top_bar(self, mx: int) -> None:
        if mx < 8:
            return
        try:
            bar = self.scr.subwin(1, mx, 0, 0)
        except curses.error:
            return
        bar.bkgd(" ", curses.color_pair(4) | curses.A_BOLD)

        fname = os.path.basename(self.path)
        left = (f" pqview - {fname}  |  {self.data.num_rows:,}x{self.data.num_cols}  "
                f"|  {self.data._load_time:.2f}s  ")
        right = " q:quit  ?:help  /:search  ^K:cmd  g/G:top/bot  r:reload "

        self._sadd(bar, 0, 0, left[:mx], curses.color_pair(4) | curses.A_BOLD)
        rpos = mx - len(right)
        if rpos > 0 and len(left) + len(right) <= mx:
            self._sadd(bar, 0, rpos, right, curses.color_pair(4) | curses.A_BOLD)

    def _draw_table(self, my: int, mx: int) -> None:
        body_h = my - 2  # top bar + status
        if body_h < 1:
            return

        vis_cols, col_xs = self._visible_cols()
        if not vis_cols:
            return

        rnw = self._rn_width()
        dx = rnw  # data starts after row-number column

        # ── header row ────────────────────────────────────────────────
        self._sadd(self.scr, 1, 0, " " * rnw, curses.color_pair(1) | curses.A_BOLD)

        for ci, c in enumerate(vis_cols):
            w = self.data.col_width(c)
            name = self.data.columns[c]
            dtype = self.data._dtypes[c]
            inner = w - 1
            if inner >= len(name) + len(dtype) + 3:
                hdr = f" {name} [{dtype}]"
            else:
                hdr = f" {name}"[:w]
            attr = curses.color_pair(2) | curses.A_BOLD if c == self.sel_col else curses.color_pair(1) | curses.A_BOLD
            self._sadd(self.scr, 1, dx + col_xs[ci], hdr.ljust(w), attr)

        # ── separator ─────────────────────────────────────────────────
        sep_w = min(dx + sum(self.data.col_width(c) for c in vis_cols), mx)
        self._sadd(self.scr, 2, 0, "-" * sep_w, curses.color_pair(5))

        # ── data rows ─────────────────────────────────────────────────
        tbl_y = 3
        tbl_h = body_h - 1
        if tbl_h < 1:
            return

        row_start = self.row
        row_end = min(row_start + tbl_h, self.data.num_rows)
        rows = self.data.get_row_batch(row_start, row_end - row_start)

        for li, row_vals in enumerate(rows):
            y = tbl_y + li
            if y >= my - 1:
                break
            gr = row_start + li

            # Row number
            rn = str(gr).rjust(rnw - 1)
            self._sadd(self.scr, y, 0, f" {rn}", curses.color_pair(5))

            for ci, c in enumerate(vis_cols):
                x = dx + col_xs[ci]
                w = self.data.col_width(c)
                val = row_vals[c][:w - 1] if c < len(row_vals) else ""

                attr = curses.A_NORMAL
                if c == self.sel_col:
                    attr = curses.color_pair(2)
                elif gr in self.search_results:
                    attr = curses.color_pair(3)

                if not val:
                    attr = curses.color_pair(6)
                    val = "."

                self._sadd(self.scr, y, x, f" {val}".ljust(w)[:w], attr)

        # ── scrollbar ─────────────────────────────────────────────────
        if self.data.num_rows > tbl_h and mx > 1:
            pct = self.row / max(1, self.data.num_rows - tbl_h)
            sb_h = max(1, tbl_h - 2)
            sb_pos = int(pct * (sb_h - 1)) if sb_h > 1 else 0
            for i in range(sb_h):
                y = tbl_y + 1 + i
                ch = "#" if i == sb_pos else "|"
                self._sadd(self.scr, y, mx - 1, ch, curses.color_pair(5))

    def _draw_status_raw(self, text: str) -> None:
        my, mx = self.scr.getmaxyx()
        if my < 2 or mx < 8:
            return
        y = my - 1
        try:
            line = self.scr.subwin(1, mx, y, 0)
            line.bkgd(" ", curses.color_pair(4))
            self._sadd(line, 0, 0, text[:mx - 1], curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass

    def _draw_help(self, my: int, mx: int) -> None:
        lines = [
            " pqview — Key bindings ",
            "",
            " Navigation              Selection & Search",
            " ──────────              ──────────────────",
            " ↑↓←→ / hjkl  scroll     s/S  cycle selected column",
            " PgUp/PgDn    page       /    search in selected column",
            " g / G        top/bot    n/N  next / previous match",
            " 0 / $        first/last Esc  cancel search",
            " Ctrl+K        command prompt",
            " c              jump col",
            "",
            " Commands:",
            "   goto end     go to last row",
            "   goto top     go to first row",
            "",
            " r  reload file",
            " ?  toggle help",
            " q  quit",
        ]
        w = min(60, mx - 4)
        h = len(lines) + 2
        bx = max(0, (mx - w) // 2)
        by = max(0, (my - h) // 2)

        try:
            win = self.scr.subwin(h, w, by, bx)
        except curses.error:
            return
        win.bkgd(" ", curses.color_pair(4) | curses.A_BOLD)
        try:
            win.box()
        except curses.error:
            pass
        for i, line in enumerate(lines):
            if i >= h - 2:
                break
            self._sadd(win, i + 1, 2, line[:w - 4])


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pqview.py <file.parquet|file.csv>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        curses.wrapper(lambda stdscr: PqView(stdscr, path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
