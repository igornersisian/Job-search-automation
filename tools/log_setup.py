"""Shared logging setup so modules don't each re-declare the same format.

`get_logger` configures the root logger once (basicConfig is idempotent — only
the first call wins) and returns a named logger, replacing the identical
basicConfig + getLogger block that was copy-pasted across the tools.
"""

import logging

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format=_FORMAT, level=logging.INFO)
    return logging.getLogger(name)
