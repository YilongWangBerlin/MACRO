# eval/edit_distance.py
from typing import Optional


def levenshtein_distance(a: str, b: str) -> int:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)

    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # DP with rolling arrays
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,         # delete
                cur[j - 1] + 1,      # insert
                prev[j - 1] + cost,  # substitute
            )
        prev = cur
    return prev[-1]


def safe_edit_distance(a: str, b: str) -> Optional[float]:
    """
    Normalized edit distance: levenshtein(a,b) / abs(len(a)).
    - a: reference/source string (denominator)
    - returns None if len(a)==0 or any exception
    """
    try:
        a = "" if a is None else str(a)
        b = "" if b is None else str(b)

        denom = abs(len(a))
        if denom == 0:
            return None

        d = levenshtein_distance(a, b)
        return float(d) / float(denom)
    except Exception:
        return None