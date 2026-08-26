"""Lane rope detection.

Lane ropes are the one thing every race video has in common: long, straight,
evenly spaced lines of floats with a repeating pattern, lying on the water.
This module finds them in a frame, reads the float pattern along each one,
and locates the places where the pattern changes (the 5 m and 15 m marks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Rope:
    # line through the rope in full-frame pixel coordinates: y = a * x + b
    a: float
    b: float
    x_min: float
    x_max: float
    score: float = 0.0
    xs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dark: np.ndarray = field(default_factory=lambda: np.zeros(0))     # dark-float density along x
    bright: np.ndarray = field(default_factory=lambda: np.zeros(0))   # bright-float density along x
    transitions: list = field(default_factory=list)                   # [(x, kind)], kind: "alt->solid" or "solid->alt"
    unknown: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # columns hidden by splash
    float_x0: float = -1.0                                            # where the floats begin and end
    float_x1: float = -1.0

    def y_at(self, x: float) -> float:
        return self.a * x + self.b


def water_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 85) & (h <= 115) & (s > 60) & (v > 80)).astype(np.uint8)


def water_body_mask(rgb: np.ndarray) -> np.ndarray:
    """Stricter water: excludes shaded deck, which can look blue-grey."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 88) & (h <= 110) & (s > 90) & (v > 110)).astype(np.uint8)


def pool_rows(water: np.ndarray, min_frac: float = 0.35) -> tuple[int, int]:
    """Top and bottom image rows of the pool (where most pixels are water)."""
    frac = water.mean(axis=1)
    rows = np.where(frac > min_frac)[0]
    if rows.size == 0:
        return 0, water.shape[0] - 1
    return int(rows[0]), int(rows[-1])


def _line_profile(img: np.ndarray, a: float, b: float, x0: int, x1: int, half: int = 3) -> np.ndarray:
    """Max over a thin band around the line, per column, for a 2-D array."""
    xs = np.arange(x0, x1)
    ys = a * xs + b
    out = np.zeros(len(xs), np.float32)
    h = img.shape[0]
    for k in range(-half, half + 1):
        yy = np.clip(np.round(ys + k).astype(int), 0, h - 1)
        out = np.maximum(out, img[yy, xs].astype(np.float32))
    return out


def detect_ropes(rgb: np.ndarray, debug: bool = False) -> list[Rope]:
    """Find lane ropes in a frame. Returns ropes sorted top to bottom."""
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    water = water_mask(rgb)
    top, bottom = pool_rows(water)

    # rope floats: dark floats, yellow floats, and pale floats that are not blue
    dark = (val < 90)
    yellow = (hue >= 12) & (hue <= 45) & (sat > 70) & (val > 90)
    pale = (sat < 60) & (val > 150) & ~((hue >= 85) & (hue <= 115) & (sat > 40))
    floats = (dark | yellow) & (water == 0)
    floats[:top, :] = False
    floats[bottom + 1:, :] = False
    # thin the mask vertically so blobs (swimmers, splash) contribute less
    m = floats.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))

    lines = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=120, minLineLength=int(W * 0.18), maxLineGap=int(W * 0.05))
    cands = []
    if lines is not None:
        # OpenCV builds differ on the shape here: (N, 1, 4) or (N, 4)
        for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
            if x1 == x0:
                continue
            a = (y1 - y0) / float(x1 - x0)
            if abs(np.degrees(np.arctan(a))) > 30:
                continue
            b = y0 - a * x0
            cands.append((a, b, min(x0, x1), max(x0, x1), np.hypot(x1 - x0, y1 - y0)))
    if not cands:
        return []

    # merge segments that lie on the same line (compare y at the image centre and slope)
    cands.sort(key=lambda c: c[0] * (W / 2) + c[1])
    groups = []
    for c in cands:
        yc = c[0] * (W / 2) + c[1]
        placed = False
        for g in groups:
            gy = g["a"] * (W / 2) + g["b"]
            if abs(yc - gy) < 12 and abs(c[0] - g["a"]) < 0.06:
                w_old, w_new = g["w"], c[4]
                g["a"] = (g["a"] * w_old + c[0] * w_new) / (w_old + w_new)
                g["b"] = (g["b"] * w_old + c[1] * w_new) / (w_old + w_new)
                g["x0"] = min(g["x0"], c[2])
                g["x1"] = max(g["x1"], c[3])
                g["w"] += w_new
                placed = True
                break
        if not placed:
            groups.append({"a": c[0], "b": c[1], "x0": c[2], "x1": c[3], "w": c[4]})

    ropes = []
    dark_img = dark.astype(np.uint8)
    yellow_img = yellow.astype(np.uint8)
    pale_img = pale.astype(np.uint8)
    foam_img = ((sat < 70) & (val > 200)).astype(np.uint8)
    for g in groups:
        a, b = g["a"], g["b"]
        # extend across the whole width but keep the measured extent
        xs = np.arange(0, W)
        ys = a * xs + b
        inside = (ys >= 0) & (ys < H)
        if inside.sum() < W * 0.3:
            continue
        x0, x1 = int(xs[inside][0]), int(xs[inside][-1])
        d = _line_profile(dark_img, a, b, x0, x1)
        y_ = _line_profile(yellow_img, a, b, x0, x1)
        p = _line_profile(pale_img, a, b, x0, x1)
        # a rope has floats (yellow or pale) along most of its length, with a
        # periodic dark/bright pattern; splash has no periodic structure
        coverage = np.maximum(np.maximum(d, y_), p)
        k = 41
        cov = np.convolve(coverage, np.ones(k) / k, mode="same")
        frac = float((cov > 0.35).mean())
        if frac < 0.45:
            continue
        rope = Rope(a=a, b=b, x_min=x0, x_max=x1, score=frac * g["w"])
        rope.xs = np.arange(x0, x1)
        win = 61
        rope.dark = np.convolve(d, np.ones(win) / win, mode="same")
        rope.bright = np.convolve(np.maximum(y_, p), np.ones(win) / win, mode="same")
        # where the floats actually are (the rope ends at the walls)
        on = cov > 0.35
        runs = _runs(on)
        long_on = [r for r in runs if r[2] and r[1] - r[0] >= 0.1 * W]
        if long_on:
            rope.float_x0 = float(rope.xs[long_on[0][0]])
            rope.float_x1 = float(rope.xs[min(long_on[-1][1], len(rope.xs) - 1)])
        # columns where splash covers the rope: white water above or below it
        foam = foam_img
        band = np.zeros(len(rope.xs), np.float32)
        for k in (-14, -9, 9, 14):
            yy = np.clip(np.round(rope.a * rope.xs + rope.b + k).astype(int), 0, H - 1)
            band = np.maximum(band, foam[yy, rope.xs].astype(np.float32))
        rope.unknown = np.convolve(band, np.ones(win) / win, mode="same") > 0.25
        ropes.append(rope)

    # ropes lie on the water: require water just above and just below the line
    kept = []
    for r in ropes:
        xs = np.arange(int(r.x_min), int(r.x_max), 4)
        ys = r.a * xs + r.b
        off = max(int(0.02 * H), 8)
        up = np.clip((ys - off).astype(int), 0, H - 1)
        dn = np.clip((ys + off).astype(int), 0, H - 1)
        w_up = water[up, xs].mean()
        w_dn = water[dn, xs].mean()
        if min(w_up, w_dn) < 0.35 or (w_up + w_dn) / 2 < 0.5:
            continue
        kept.append(r)
    ropes = kept

    # ropes are parallel in the world, so they meet at one vanishing point in
    # the image; discard lines that disagree with the consensus
    ropes = _vanishing_point_filter(ropes, W, H)
    ropes = _no_crossing_filter(ropes, W)

    ropes.sort(key=lambda r: r.y_at(W / 2))
    # drop near-duplicates (two edges of the same rope)
    merged = []
    for r in ropes:
        if merged and abs(r.y_at(W / 2) - merged[-1].y_at(W / 2)) < 0.35 * _median_spacing(ropes, W):
            if r.score > merged[-1].score:
                merged[-1] = r
            continue
        merged.append(r)
    for r in merged:
        r.transitions = find_transitions(r)
    return merged


def _vanishing_point_filter(ropes: list[Rope], W: int, H: int) -> list[Rope]:
    if len(ropes) < 3:
        return ropes
    # intersection of each pair of lines
    pts = []
    for i in range(len(ropes)):
        for j in range(i + 1, len(ropes)):
            a1, b1, a2, b2 = ropes[i].a, ropes[i].b, ropes[j].a, ropes[j].b
            if abs(a1 - a2) < 1e-4:
                continue
            x = (b2 - b1) / (a1 - a2)
            pts.append((x, a1 * x + b1, i, j))
    if not pts:
        return ropes
    xs = np.array([p[0] for p in pts])
    # consensus: the x with most intersections within a tolerance (lines are
    # nearly parallel, so the vanishing point is far off to one side)
    best, best_n = None, 0
    for cx in xs:
        n = int((np.abs(xs - cx) < max(0.15 * abs(cx - W / 2), 3 * W)).sum())
        if n > best_n:
            best, best_n = cx, n
    if best is None:
        return ropes
    tol = max(0.15 * abs(best - W / 2), 3 * W)
    votes = np.zeros(len(ropes))
    for x, y, i, j in pts:
        if abs(x - best) < tol:
            votes[i] += 1
            votes[j] += 1
    keep = [r for r, v in zip(ropes, votes) if v >= max(1, 0.4 * (len(ropes) - 1))]
    return keep if len(keep) >= 2 else ropes


def _median_spacing(ropes: list[Rope], W: int) -> float:
    ys = sorted(r.y_at(W / 2) for r in ropes)
    if len(ys) < 2:
        return 60.0
    gaps = np.diff(ys)
    gaps = gaps[gaps > 8]
    return float(np.median(gaps)) if gaps.size else 60.0


def _runs(flags) -> list[tuple[int, int, object]]:
    """Consecutive runs of equal values: [(start, end_exclusive, value)]."""
    runs = []
    start = 0
    n = len(flags)
    for i in range(1, n + 1):
        if i == n or flags[i] != flags[start]:
            runs.append((start, i, flags[start]))
            start = i
    return runs


def _no_crossing_filter(ropes: list[Rope], W: int) -> list[Rope]:
    """Ropes never cross inside the picture. When two lines do, keep the
    stronger one."""
    ropes = sorted(ropes, key=lambda r: -r.score)
    kept = []
    for r in ropes:
        ok = True
        for k in kept:
            if abs(r.a - k.a) < 1e-6:
                continue
            x = (k.b - r.b) / (r.a - k.a)
            if -0.2 * W <= x <= 1.2 * W:
                ok = False
                break
        if ok:
            kept.append(r)
    return kept


def find_transitions(rope: Rope, min_run: int = 120) -> list[tuple[float, str]]:
    """Where the float pattern switches between alternating (dark floats
    present) and solid (no dark floats), ignoring stretches hidden by splash.
    Returns [(x, kind)]."""
    if rope.dark.size == 0:
        return []
    state = np.where(rope.dark > 0.12, 1, 0)
    if rope.unknown.size == len(state):
        state = np.where(rope.unknown, -1, state)
    # outside the float extent nothing is known
    if rope.float_x0 >= 0:
        state[rope.xs < rope.float_x0] = -1
        state[rope.xs > rope.float_x1] = -1
    runs = [r for r in _runs(state) if r[2] != -1 and r[1] - r[0] >= min_run]
    out = []
    for (s0, e0, k0), (s1, e1, k1) in zip(runs[:-1], runs[1:]):
        if k0 == k1:
            continue
        gap = rope.xs[min(s1, len(rope.xs) - 1)] - rope.xs[min(e0, len(rope.xs) - 1)]
        if gap > 120:
            continue  # the change happened somewhere under splash; too vague
        x = float(rope.xs[min(e0, len(rope.xs) - 1)] + rope.xs[min(s1, len(rope.xs) - 1)]) / 2
        out.append((x, "alt->solid" if k0 == 1 else "solid->alt"))
    return out


def mark_lines(ropes: list[Rope], min_ropes: int = 3, tol: float = 10.0) -> list[dict]:
    """Group transitions that line up across ropes into straight lines.

    A colour change at 15 m happens on every rope at once, so the points
    form a straight line across the pool (perpendicular to the ropes in the
    world). Returns [{"pts": [(x, y, rope_index)], "line": (a, b), "kinds"}].
    """
    pts = []
    for i, r in enumerate(ropes):
        for x, kind in r.transitions:
            pts.append((x, r.y_at(x), i, kind))
    if len(pts) < min_ropes:
        return []
    remaining = list(range(len(pts)))
    lines = []
    while len(remaining) >= min_ropes:
        best = None
        for i in remaining:
            for j in remaining:
                if j <= i or pts[i][2] == pts[j][2]:
                    continue
                (x0, y0), (x1, y1) = pts[i][:2], pts[j][:2]
                if abs(y1 - y0) < 1e-6:
                    continue
                # line as x = c * y + d (mark lines are steep-ish in a side view)
                c = (x1 - x0) / (y1 - y0)
                d = x0 - c * y0
                members = [k for k in remaining if abs(pts[k][0] - (c * pts[k][1] + d)) < tol]
                ropes_hit = {pts[k][2] for k in members}
                if len(ropes_hit) >= min_ropes and (best is None or len(ropes_hit) > best[0]):
                    best = (len(ropes_hit), c, d, members)
        if best is None:
            break
        _, c, d, members = best
        # one point per rope
        per_rope = {}
        for k in members:
            per_rope.setdefault(pts[k][2], pts[k])
        lines.append({"pts": [(p[0], p[1], p[2]) for p in per_rope.values()], "line": (c, d),
                      "kinds": [p[3] for p in per_rope.values()]})
        remaining = [k for k in remaining if k not in members]
    return lines


def draw_ropes(rgb: np.ndarray, ropes: list[Rope], marks: list[dict] | None = None) -> np.ndarray:
    img = rgb.copy()
    for m in marks or []:
        c, d = m["line"]
        ys = [p[1] for p in m["pts"]]
        y0, y1 = min(ys) - 40, max(ys) + 40
        cv2.line(img, (int(c * y0 + d), int(y0)), (int(c * y1 + d), int(y1)), (255, 255, 255), 2)
    for i, r in enumerate(ropes):
        x0, x1 = int(r.x_min), int(r.x_max)
        cv2.line(img, (x0, int(r.y_at(x0))), (x1, int(r.y_at(x1))), (255, 0, 255), 1)
        if r.float_x0 >= 0:
            fx0, fx1 = int(r.float_x0), int(r.float_x1)
            cv2.line(img, (fx0, int(r.y_at(fx0))), (fx1, int(r.y_at(fx1))), (255, 0, 255), 3)
        cv2.putText(img, str(i), (x0 + 5, int(r.y_at(x0)) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        for x, kind in r.transitions:
            y = int(r.y_at(x))
            color = (255, 255, 0) if kind == "alt->solid" else (0, 255, 255)
            cv2.circle(img, (int(x), y), 9, color, -1)
            cv2.circle(img, (int(x), y), 9, (0, 0, 0), 2)
    return img
