import operator
from typing import List, Tuple

import numpy


def __cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def __convex_hull(points):
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and __cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and __cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def __slope(p, q):
    dx = (q[0] - p[0])
    return (q[1] - p[1]) / dx if dx != 0 else 0


def __find_minmaximizer(hull, point, minimize):
    best_i = 0
    comp = operator.le if minimize else operator.ge
    best = 2 ** 31 if minimize else -1
    for i, q in enumerate(hull):
        if point[0] == q[0]:
            continue

        if comp(__slope(q, point), best):
            best_i = i
            best = __slope(q, point)
    return best_i


def __maximal(vector, error, index, debug=False):
    sa = (vector[index], index + error)
    sb = (vector[index], index - error)
    sc = (vector[index + 1], index + 1 - error)
    sd = (vector[index + 1], index + 1 + error)

    def get_segment(absolute_coordinates=True):
        sl = (__slope(sa, sc) + __slope(sb, sd)) / 2.
        a = (sc[0] - sa[0]) * (sd[1] - sb[1]) - (sc[1] - sa[1]) * (sd[0] - sb[0])
        if a == 0:
            a = __slope(sb, sd)
        b = ((sb[0] - sa[0]) * (sd[1] - sb[1]) - (sb[1] - sa[1]) * (sd[0] - sb[0])) / a
        i_x = sa[0] + b * (sc[0] - sa[0])
        i_y = sa[1] + b * (sc[1] - sa[1])
        ic = i_y - i_x * sl if absolute_coordinates else i_y - (i_x - vector[index]) * sl
        return sl, ic

    L = [sa, sd]
    U = [sb, sc]
    y = index + 2
    pts = [(vector[index], index), (vector[index + 1], index + 1)]
    L_slope = __slope(sa, sc)
    U_slope = __slope(sb, sd)

    while y < len(vector):
        x = vector[y]
        pts.append((x, y))

        if not (L_slope * (x - sc[0]) + sc[1] - error <= y <= U_slope * (x - sd[0]) + sd[1] + error):
            break

        s = (x, y + error)
        j = __find_minmaximizer(U, s, True)
        if __slope(U[j], s) <= U_slope:
            sb = U[j]
            sd = s
            U_slope = __slope(sb, sd)
            U = U[j:]
        while len(L) >= 2 and __cross(L[-2], L[-1], s) <= 0:
            L.pop()
        L.append(s)

        s = (x, y - error)
        j = __find_minmaximizer(L, s, False)
        if __slope(L[j], s) >= L_slope:
            sa = L[j]
            sc = s
            L_slope = __slope(sa, sc)
            L = L[j:]
        while len(U) >= 2 and __cross(U[-2], U[-1], s) >= 0:
            U.pop()
        U.append(s)
        assert len(U) >= 2
        assert len(L) >= 2

        if y == 500 and debug:
            ex = vector[index:y + 1]
            sl, ic = get_segment()
            val = ex * sl + ic
        y += 1
    sl, ic = get_segment(False)
    return y, sl, ic


def pla(vector: numpy.ndarray, error: float = 32) -> List[Tuple[int, int, float, float]]:
    i = 0
    nr_segments = 0
    e = 0
    segments = []

    while i + 2 < len(vector):
        start = i
        (i, sl, ic) = __maximal(vector, error, start)
        segments.append((start, i, sl, ic))
        for j in range(start, i):
            if j > 0 and vector[j] == vector[j-1]:
                continue
            e += abs(sl * (vector[j] - vector[start]) + ic - j) > error
        nr_segments += 1

    return segments
