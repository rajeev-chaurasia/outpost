"""The failure ladder's rung.

Selection logic (degrade.py's actual job) is added in phase 5.
"""

from enum import IntEnum


class Rung(IntEnum):
    """Which of the five failure-ladder behaviors an answer used."""

    FULL = 1
    PARTIAL = 2
    ACTION_DECLINED = 3
    PROVIDER_FALLBACK = 4
    REFUSED = 5
