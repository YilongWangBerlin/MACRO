from typing import Tuple


def levenshtein_distance(a: str, b: str) -> int:
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
                prev[j] + 1,      # delete
                cur[j - 1] + 1,   # insert
                prev[j - 1] + cost,  # substitute
            )
        prev = cur
    return prev[-1]


def reward_edit(x_hat: str, x: str) -> float:
    if x is None:
        return 0.0
    if len(x) == 0:
        return 0.0
    d = levenshtein_distance(x_hat, x)
    r = 1.0 - (d / float(len(x)))
    # allow negative if very different, or clamp if you prefer
    return float(r)
