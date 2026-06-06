"""Small shared helpers for the source scrapers' normalise_* functions."""

from __future__ import annotations


def format_salary(lo, hi, *, currency: str = "USD", sep: str = "-", suffix: str = "") -> str:
    """Build a salary string shared by the scrapers.

    Returns '' when there's no low bound. With both bounds: '<sym><lo><sep><sym><hi>';
    low only: '<sym><lo>+'. `suffix` is appended verbatim (e.g. ' Yearly' or
    '/Hour') and the result is stripped, so an empty period/unit leaves no
    trailing space. Numbers are thousands-formatted as given — cast to int at
    the call site if the source returns floats you don't want decimals on.
    """
    if not lo:
        return ""
    symbol = "$" if currency == "USD" else f"{currency} "
    if hi:
        body = f"{symbol}{lo:,}{sep}{symbol}{hi:,}"
    else:
        body = f"{symbol}{lo:,}+"
    return f"{body}{suffix}".strip()
