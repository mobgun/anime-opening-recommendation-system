"""Make stdout able to carry the characters this project actually prints.

Song and anime titles come from AnimeThemes verbatim, so they contain characters
that no 8-bit codepage covers — `Coda〜Death note`, `R★O★C★K★S`, `Tokyo Ghoul √A`.
On Windows `sys.stdout.encoding` follows the console codepage (cp1251 here), and
`print` of such a title raises UnicodeEncodeError, which killed the CLI on 7 of the
621 themes in the reference corpus before this existed.

Two steps, because either one alone is not enough: switching the process's streams
to UTF-8 stops the crash, and switching the console codepage is what makes the
result render as the character rather than as mojibake. `errors="replace"` is the
last line of defence — a terminal that still cannot show a glyph gets `?` and the
command completes.
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    """Idempotent; safe when stdout is a pipe, a file or a test capture buffer."""
    if sys.platform == "win32":
        try:
            import ctypes

            # 65001 = UTF-8. Fails harmlessly when there is no console attached.
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # pytest's capture object, or an already-wrapped stream
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
