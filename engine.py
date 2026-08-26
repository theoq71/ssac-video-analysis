"""Automatic race analysis.

Input: a race video and one tap on the swimmer in the start frame.
Output: everything the coach wants to know about the lap, with no other
human input.

Stages
------
1. scan():   per frame, find the lane ropes, the colour-change marks on
             them, the camera motion, and the white water in the swimmer's
             lane. Cheap enough to run on every frame.
2. resolve(): turn the per-frame observations into pool-fixed landmarks
             (walls, 5 m, 15 m, 35 m, 45 m marks), a metres-along-the-lane
             position track for the swimmer, and the split crossings.
3. metrics:  handed to metrics.analyze() together with the stroke rhythm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

from ropes import Rope, detect_ropes, mark_lines, water_mask, water_body_mask

WORK_W = 1280


# ---------------------------------------------------------------------------
# per-frame observations
# ---------------------------------------------------------------------------

@dataclass
class FrameObs:
    frame: int
    dx: float = 0.0            # camera shift from previous frame (pixels, work scale)
    dy: float = 0.0
    lane_top: Optional[tuple] = None    # (a, b) of the rope above the lane
    lane_bot: Optional[tuple] = None    # (a, b) of the rope below the lane
    lane_matched: int = 0                # how many of the two ropes were seen this frame
    spacing: float = 0.0                 # rope spacing at image centre
    vp_x: Optional[float] = None         # vanishing point x of the ropes
    marks: list = field(default_factory=list)      # x of mark lines where they cross the lane centre
    mark_pts: list = field(default_factory=list)   # raw mark line points (for pictures)
    wall_x: Optional[float] = None       # x where the water ends on the right, if inside the picture
    wall_side: int = 0
    wall_left: Optional[float] = None    # x where the water begins on the left, if inside the picture
    pool_x0: float = 0.0                 # extent of the pool along the lane in this frame
    pool_x1: float = 0.0
    tracker_state: Optional[tuple] = None
    foam: Optional[np.ndarray] = None    # foam fraction per column (work scale)
    foam_xs: Optional[np.ndarray] = None
    core: Optional[np.ndarray] = None    # foam fraction in the middle third of the lane
    edge_left: Optional[float] = None    # leftmost / rightmost sustained white water
    edge_right: Optional[float] = None
    centroid: Optional[float] = None
    foam_amount: float = 0.0


def _shift(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    a = np.float32(cv2.resize(prev_gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    b = np.float32(cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return dx * 4.0, dy * 4.0


def _line_shift(line: tuple, dx: float, dy: float) -> tuple:
    a, b = line
    return (a, b + dy - a * dx)


def _rope_spacing_at(ropes: list[Rope], x: float) -> float:
    ys = sorted(r.y_at(x) for r in ropes)
    gaps = np.diff(ys)
    gaps = gaps[gaps > 6]
    return float(np.median(gaps)) if gaps.size else 60.0


def _vanishing_x(ropes: list[Rope]) -> Optional[float]:
    if len(ropes) < 2:
        return None
    xs = []
    for i in range(len(ropes)):
        for j in range(i + 1, len(ropes)):
            a1, b1, a2, b2 = ropes[i].a, ropes[i].b, ropes[j].a, ropes[j].b
            if abs(a1 - a2) > 1e-4:
                xs.append((b2 - b1) / (a1 - a2))
    return float(np.median(xs)) if xs else None


def wall_side_from_water(water: np.ndarray) -> int:
    """+1 if the pool ends (wall) inside the picture on the right, -1 on the
    left, 0 if the water reaches both edges."""
    H, W = water.shape
    rows = np.where(water.mean(axis=1) > 0.3)[0]
    if rows.size == 0:
        return 0
    right_in = 0
    left_in = 0
    for y in rows[::4]:
        cols = np.where(water[y] > 0)[0]
        if cols.size == 0:
            continue
        if cols[-1] < W - 12:
            right_in += 1
        if cols[0] > 12:
            left_in += 1
    n = max(len(rows[::4]), 1)
    if right_in / n > 0.5 and right_in >= left_in:
        return 1
    if left_in / n > 0.5:
        return -1
    return 0


def water_top_boundary(water: np.ndarray, x0: int, x1: int, step: int = 6) -> list[tuple[int, int]]:
    """For columns x0..x1, the first row where the water starts for good."""
    H, W = water.shape
    below = max(int(0.09 * H), 30)
    above = max(int(0.06 * H), 20)
    out = []
    for x in range(max(x0, 6), min(x1, W - 7), step):
        band = water[:, x - 6:x + 7].mean(axis=1).astype(np.float32)
        cs = np.concatenate([[0.0], np.cumsum(band)])
        best, best_y = 0.0, None
        for y in range(above, H - below):
            wb = (cs[y + below] - cs[y]) / below
            wa = (cs[y] - cs[y - above]) / above
            score = wb - wa
            if wb > 0.5 and score > best:
                best, best_y = score, y
        if best_y is not None:
            out.append((x, best_y))
    return out


def wall_line_near(water: np.ndarray, tx: float, ty: float) -> Optional[tuple]:
    """The wall (deck edge) near a point above the water, as y = a x + b.

    Fitted through the columns where the water starts highest, so blocks
    overhanging the edge do not pull it down."""
    pts = water_top_boundary(water, int(tx - 260), int(tx + 260))
    pts = [(x, y) for x, y in pts if y > ty]
    if len(pts) < 6:
        return None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    keep = []
    for k in range(len(pts)):
        near = np.abs(xs - xs[k]) <= 45
        if ys[k] <= ys[near].min() + 4:
            keep.append(k)
    if len(keep) < 3:
        return None
    a, b = np.polyfit(xs[keep], ys[keep], 1)
    return float(a), float(b)


def lane_from_tap(ropes: list[Rope], tap: tuple[float, float], W: int, H: int,
                  water: Optional[np.ndarray] = None) -> Optional[tuple[Rope, Rope, int]]:
    """Pick the two ropes that bound the tapped swimmer's lane.

    A swimmer on the block stands above the wall. The wall is the upper
    boundary of the water near the tap; blocks overhang it, so the boundary
    is fitted through the columns where the water starts highest. Each rope
    meets the wall somewhere along that line, the lanes are the stretches
    between those points, and the tap dropped straight down onto the wall
    lands in one of them. A tap on the water uses the rope lines directly.
    Returns (top rope, bottom rope, wall side).
    """
    if len(ropes) < 2:
        return None
    order = sorted(ropes, key=lambda r: r.y_at(W / 2))
    tx, ty = tap
    wall_side = wall_side_from_water(water) if water is not None else 0

    def bracket(x, y):
        for r0, r1 in zip(order[:-1], order[1:]):
            if r0.y_at(x) <= y <= r1.y_at(x):
                return r0, r1
        return None

    # tap on the water: the ropes around it
    if water is not None and water[int(np.clip(ty, 0, H - 1)), int(np.clip(tx, 0, W - 1))] and bracket(tx, ty):
        hit = bracket(tx, ty)
        return hit[0], hit[1], wall_side

    wall = wall_line_near(water, tx, ty) if water is not None else None
    if wall is not None:
        a, b = wall
        if True:
            if True:
                # where each rope meets the wall
                pts_w = []
                for r in order:
                    if abs(r.a - a) < 1e-6:
                        continue
                    xw = (b - r.b) / (r.a - a)
                    pts_w.append((xw, r))
                pts_w.sort(key=lambda p: p[0])
                # lanes beyond the outermost detected ropes: extrapolate one
                # more rope on each side from the neighbouring spacing
                if len(pts_w) >= 3:
                    (xa, ra), (xb, rb) = pts_w[0], pts_w[1]
                    virt = Rope(a=2 * ra.a - rb.a, b=2 * ra.b - rb.b, x_min=0, x_max=W)
                    pts_w.insert(0, (2 * xa - xb, virt))
                    (xa, ra), (xb, rb) = pts_w[-1], pts_w[-2]
                    virt = Rope(a=2 * ra.a - rb.a, b=2 * ra.b - rb.b, x_min=0, x_max=W)
                    pts_w.append((2 * xa - xb, virt))
                # the tap dropped onto the wall
                x_tap = tx
                for (x0, r0), (x1, r1) in zip(pts_w[:-1], pts_w[1:]):
                    if x0 <= x_tap <= x1:
                        top, bot = sorted((r0, r1), key=lambda r: r.y_at(W / 2))
                        return top, bot, wall_side
    hit = bracket(tx, ty)
    if hit:
        return hit[0], hit[1], wall_side
    return None


def _match_rope(pred: tuple, ropes: list[Rope], W: int, tol: float) -> Optional[Rope]:
    a, b = pred
    yc = a * (W / 2) + b
    best, best_d = None, tol
    for r in ropes:
        d = abs(r.y_at(W / 2) - yc)
        if d < best_d:
            best, best_d = r, d
    return best


def lane_strip_profile(rgb: np.ndarray, top: tuple, bot: tuple, hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foam and body fraction per column inside the lane (between the ropes,
    trimmed so the ropes themselves are excluded)."""
    H, W = rgb.shape[:2]
    xs = np.arange(0, W, 2, dtype=np.float32)
    y_top = top[0] * xs + top[1]
    y_bot = bot[0] * xs + bot[1]
    inner_top = y_top + (y_bot - y_top) * 0.14
    inner_bot = y_bot - (y_bot - y_top) * 0.14
    n = 24
    ts = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    map_y = inner_top[None, :] + ts * (inner_bot - inner_top)[None, :]
    map_x = np.repeat(xs[None, :], n, axis=0).astype(np.float32)
    map_y = map_y.astype(np.float32)
    valid = (map_y >= 0) & (map_y < H)
    strip = cv2.remap(hsv, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(100, 200, 0))
    s = strip[..., 1].astype(np.int16)
    v = strip[..., 2].astype(np.int16)
    h = strip[..., 0].astype(np.int16)
    white = (s < 70) & (v > 200) & valid
    foam = white.mean(axis=0)
    # the middle third of the lane: a neighbour's splash spilling over a
    # rope rarely reaches it, the swimmer's own wake always does
    core = white[n // 3: 2 * n // 3].mean(axis=0)
    return xs, foam.astype(np.float32), core.astype(np.float32)


def _runs_bool(flags):
    runs = []
    start = 0
    n = len(flags)
    for i in range(1, n + 1):
        if i == n or flags[i] != flags[start]:
            runs.append((start, i, bool(flags[start])))
            start = i
    return runs


def _leading_edge(xs: np.ndarray, score: np.ndarray, direction: int, min_score: float, min_run: int) -> Optional[float]:
    on = score >= min_score
    order = range(len(on)) if direction < 0 else range(len(on) - 1, -1, -1)
    run = 0
    start = None
    for i in order:
        if on[i]:
            if run == 0:
                start = i
            run += 1
            if run >= min_run:
                return float(xs[start])
        else:
            run = 0
    return None


class RopeSetTracker:
    """Follows every visible rope from frame to frame and remembers which
    pair of them bounds the swimmer's lane.

    Lines are kept sorted top to bottom. Each frame, all lines are moved by
    the camera shift, then snapped to the ropes actually detected. Ropes that
    were not seen are carried along for a while; new ropes are slotted into
    the order. The lane is "between line k and line k+1", and k is adjusted
    whenever lines are inserted or dropped above it.
    """

    def __init__(self, ropes: list[Rope], lane_top: tuple, lane_bot: tuple, W: int, H: int = 0):
        self.W = W
        self.H = H or int(W * 0.5625)
        lines = sorted(((r.a, r.b) for r in ropes), key=lambda l: l[0] * W / 2 + l[1])
        # make sure the lane's own lines are in the set
        for ln in (lane_top, lane_bot):
            if not any(abs(self._yc(l) - self._yc(ln)) < 4 for l in lines):
                lines.append(ln)
        lines.sort(key=self._yc)
        self.lines = [{"line": l, "missed": 0} for l in lines]
        self.k = min(range(len(lines)), key=lambda j: abs(self._yc(lines[j]) - self._yc(lane_top)))
        self.matched_ropes = (None, None)

    def _yc(self, line):
        return line[0] * self.W / 2 + line[1]

    def predict(self, dx, dy):
        for e in self.lines:
            e["line"] = _line_shift(e["line"], dx, dy)

    def local_spacing(self) -> float:
        ys = [self._yc(e["line"]) for e in self.lines]
        if self.k + 1 < len(ys):
            return abs(ys[self.k + 1] - ys[self.k])
        gaps = np.diff(ys)
        return float(np.median(gaps)) if len(gaps) else 60.0

    def lane(self):
        top = self.lines[self.k]["line"]
        if self.k + 1 < len(self.lines):
            bot = self.lines[self.k + 1]["line"]
        else:
            sp = self.local_spacing()
            bot = (top[0], top[1] + sp)
        return top, bot

    def update(self, ropes: list[Rope]) -> int:
        W2 = self.W / 2
        ys_pred = np.array([self._yc(e["line"]) for e in self.lines], float)
        det = sorted(ropes, key=lambda r: r.y_at(W2))
        ys_det = np.array([r.y_at(W2) for r in det], float)
        gaps = np.diff(ys_pred)
        med_gap = float(np.median(gaps)) if len(gaps) else 60.0

        # 1. robust global correction: the ropes move together, so fit
        #    y_det = alpha * y_pred + beta on the most consistent pairs
        if len(ys_det) >= 2 and len(ys_pred) >= 2:
            best = None
            for j0 in range(len(ys_pred)):
                for m0 in range(len(ys_det)):
                    if abs(ys_det[m0] - ys_pred[j0]) > 0.6 * med_gap:
                        continue
                    for j1 in range(j0 + 1, len(ys_pred)):
                        for m1 in range(m0 + 1, len(ys_det)):
                            if abs(ys_det[m1] - ys_pred[j1]) > 0.6 * med_gap:
                                continue
                            dp = ys_pred[j1] - ys_pred[j0]
                            if dp < 1e-6:
                                continue
                            alpha = (ys_det[m1] - ys_det[m0]) / dp
                            if not 0.96 < alpha < 1.04:
                                continue
                            beta = ys_det[m0] - alpha * ys_pred[j0]
                            # the whole picture cannot move more than about
                            # a third of a rope spacing beyond the prediction
                            if abs(alpha * W2 + beta - W2) > 0.35 * med_gap:
                                continue
                            fit = alpha * ys_pred + beta
                            d = np.abs(fit[:, None] - ys_det[None, :])
                            inl = (d.min(axis=1) < 0.25 * med_gap).sum()
                            err = d.min(axis=1).clip(max=0.25 * med_gap).sum()
                            if best is None or inl > best[0] or (inl == best[0] and err < best[1]):
                                best = (inl, err, alpha, beta)
            if best is not None and best[0] >= 2:
                _, _, alpha, beta = best
                self.last_fit = (alpha, beta, best[0])
                for e in self.lines:
                    a, b = e["line"]
                    yc = a * W2 + b
                    e["line"] = (a * alpha, alpha * yc + beta - a * alpha * W2)
                ys_pred = np.array([self._yc(e["line"]) for e in self.lines], float)

        # 2. order-preserving alignment between predicted lines and detected
        #    ropes (dynamic programming, like sequence alignment)
        n, m = len(ys_pred), len(ys_det)
        INF = 1e9
        cost = np.full((n + 1, m + 1), INF)
        back = np.zeros((n + 1, m + 1), np.int8)
        gap = 0.7
        cost[0, 0] = 0.0
        for j in range(n + 1):
            for k in range(m + 1):
                if j == 0 and k == 0:
                    continue
                cands = []
                if j > 0:
                    cands.append((cost[j - 1, k] + gap, 1))
                if k > 0:
                    cands.append((cost[j, k - 1] + gap, 2))
                if j > 0 and k > 0:
                    d = abs(ys_pred[j - 1] - ys_det[k - 1]) / med_gap
                    if d < 0.5:
                        cands.append((cost[j - 1, k - 1] + d, 3))
                c, b = min(cands)
                cost[j, k], back[j, k] = c, b
        matched_idx = {}
        j, k = n, m
        while j > 0 or k > 0:
            b = back[j, k]
            if b == 3:
                matched_idx[j - 1] = k - 1
                j, k = j - 1, k - 1
            elif b == 1:
                j -= 1
            else:
                k -= 1
        used = set(matched_idx.values())
        for j, e in enumerate(self.lines):
            if j in matched_idx:
                r = det[matched_idx[j]]
                e["line"] = (r.a, r.b)
                e["missed"] = 0
            else:
                e["missed"] += 1
        # new ropes: insert in order if they fit the spacing pattern
        for mm, r in enumerate(det):
            if mm in used:
                continue
            yc = r.y_at(W2)
            ys = [self._yc(e["line"]) for e in self.lines]
            pos = int(np.searchsorted(ys, yc))
            if pos > 0 and yc - ys[pos - 1] < 0.55 * med_gap:
                continue
            if pos < len(ys) and ys[pos] - yc < 0.55 * med_gap:
                continue
            self.lines.insert(pos, {"line": (r.a, r.b), "missed": 0})
            if pos <= self.k:
                self.k += 1
        # drop lines that left the picture or have been missing too long,
        # but never the lane's own two lines
        keep = []
        new_k = self.k
        for j, e in enumerate(self.lines):
            yc = self._yc(e["line"])
            is_lane = j in (self.k, self.k + 1)
            gone = (yc < -0.5 * med_gap or yc > self.H + 0.5 * med_gap) and not is_lane
            stale = e["missed"] > 45 and not is_lane
            if gone or stale:
                if j < self.k:
                    new_k -= 1
                continue
            keep.append(e)
        self.lines = keep
        self.k = max(0, min(new_k, len(self.lines) - 1))
        top_m = self.k in matched_idx
        bot_m = (self.k + 1) in matched_idx
        self.matched_ropes = (det[matched_idx[self.k]] if top_m else None,
                              det[matched_idx[self.k + 1]] if bot_m else None)
        return int(top_m) + int(bot_m)


def scan(video, tap: tuple[float, float], tap_frame: int, start_frame: int, progress=None,
         end_frame: Optional[int] = None) -> dict:
    """Stage 1: per-frame observations from `start_frame` to the end.

    tap: (x, y) in full-resolution pixels of the frame `tap_frame`.
    """
    W0, H0 = video.width, video.height
    scale = WORK_W / W0
    W, H = WORK_W, int(round(H0 * scale))
    last = video.frame_count - 1 if end_frame is None else min(end_frame, video.frame_count - 1)

    # lane from the tap
    rgb = cv2.resize(video.frame(tap_frame), (W, H), interpolation=cv2.INTER_AREA)
    ropes = detect_ropes(rgb)
    picked = lane_from_tap(ropes, (tap[0] * scale, tap[1] * scale), W, H, water_mask(rgb))
    if picked is None:
        raise ValueError("Could not find the lane ropes around the tapped swimmer")
    top, bot, wall_side = picked
    lane_top, lane_bot = (top.a, top.b), (bot.a, bot.b)
    wall_start_x = None
    wall = wall_line_near(water_mask(rgb), tap[0] * scale, tap[1] * scale)
    if wall is not None:
        ca, cb = (top.a + bot.a) / 2, (top.b + bot.b) / 2
        wa, wb = wall
        if abs(ca - wa) > 1e-6:
            wall_start_x = float((wb - cb) / (ca - wa))

    obs_by_frame: dict[int, FrameObs] = {}
    prev_gray = None
    # run the tracker from the tap frame forward, and from the tap frame
    # backward to the start frame (a few frames at most)
    order = list(range(tap_frame, last + 1)) + list(range(tap_frame - 1, start_frame - 1, -1))
    tracker = None
    for n, i in enumerate(order):
        f = video.frame(i)
        if f is None:
            break
        rgb = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        o = FrameObs(frame=i)
        ropes = detect_ropes(rgb)
        if i == tap_frame or i == tap_frame - 1:
            if i == tap_frame - 1:
                prev_gray = cv2.cvtColor(cv2.resize(video.frame(tap_frame), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            tracker = RopeSetTracker(ropes if i == tap_frame else tracker_init_ropes, lane_top, lane_bot, W, H)
            if i == tap_frame:
                tracker_init_ropes = ropes
                prev_gray = gray
                dx = dy = 0.0
        if i != tap_frame:
            dx, dy = _shift(prev_gray, gray)
            o.dx, o.dy = dx, dy
            prev_gray = gray
            tracker.last_fit = None
            tracker.predict(dx, dy)
        matched = tracker.update(ropes)
        cur_top, cur_bot = tracker.lane()
        o.tracker_state = ([round(tracker._yc(e["line"])) for e in tracker.lines], tracker.k, getattr(tracker, "last_fit", None))
        spacing = tracker.local_spacing()
        m_top, m_bot = tracker.matched_ropes
        o.lane_top, o.lane_bot, o.lane_matched, o.spacing = cur_top, cur_bot, matched, spacing
        o.vp_x = _vanishing_x(ropes)

        # marks: where each mark line crosses the lane centre line
        centre = ((cur_top[0] + cur_bot[0]) / 2, (cur_top[1] + cur_bot[1]) / 2)
        for m in mark_lines(ropes):
            c, d = m["line"]
            # x = c*y + d and y = a*x + b  ->  x = c*(a*x + b) + d
            denom = 1 - c * centre[0]
            if abs(denom) < 1e-6:
                continue
            x = (c * centre[1] + d) / denom
            o.marks.append(float(x))
            o.mark_pts.append([(float(p[0]), float(p[1])) for p in m["pts"]])

        # wall: where the pool ends along the lane, if that is inside the
        # picture. The pool is the connected body of water that holds the
        # lane (ropes bridged, splash holes filled); a second pool beyond the
        # deck is a different body and does not count.
        cx = np.arange(0, W, 4)
        cy = np.clip(np.round(centre[0] * cx + centre[1]).astype(int), 0, H - 1)
        wl = water_body_mask(rgb)
        kv = max(int(0.7 * spacing), 9)
        body_mask = cv2.morphologyEx(wl, cv2.MORPH_CLOSE, np.ones((kv, 5), np.uint8))
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(body_mask, connectivity=8)
        if n_lab > 1:
            # component under the lane centre line, by votes along it
            votes = np.bincount(labels[cy, cx], minlength=n_lab)
            votes[0] = 0
            lab = int(np.argmax(votes))
            comp = (labels == lab).astype(np.uint8)
            # fill holes (splash inside the pool)
            flood = comp.copy()
            ff_mask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(flood, ff_mask, (0, 0), 1)
            cv2.floodFill(flood, ff_mask, (W - 1, 0), 1)
            cv2.floodFill(flood, ff_mask, (0, H - 1), 1)
            cv2.floodFill(flood, ff_mask, (W - 1, H - 1), 1)
            holes = (flood == 0) & (comp == 0)
            comp[holes] = 1
            wet = comp[cy, cx] > 0
        else:
            wet = wl[cy, cx] > 0
        # swimmer's white water in the lane
        xs, foam, core = lane_strip_profile(rgb, cur_top, cur_bot, hsv)
        o.foam_xs, o.core = xs, core
        run_min = int(0.06 * W / 4)
        runs = _runs_bool(wet)
        filled = wet.copy()
        for k_, (a_, b_, v_) in enumerate(runs):
            if not v_ and 0 < k_ < len(runs) - 1 and b_ - a_ < run_min:
                filled[a_:b_] = True
        runs = _runs_bool(filled)
        mid = len(cx) // 2
        pool_x0, pool_x1 = 0.0, float(W)
        cand = [r_ for r_ in runs if r_[2]]
        if cand:
            holder = [r_ for r_ in cand if r_[0] <= mid < r_[1]]
            a_, b_, _ = holder[0] if holder else max(cand, key=lambda r_: r_[1] - r_[0])
            if b_ < len(cx) - 6:
                o.wall_x, o.wall_side = float(cx[b_ - 1]), 1
                pool_x1 = float(cx[b_ - 1])
            if a_ > 6:
                o.wall_left = float(cx[a_])
                pool_x0 = float(cx[a_])
        o.pool_x0, o.pool_x1 = pool_x0, pool_x1
        o.foam = foam
        o.foam_amount = float(foam.mean())
        o.edge_left = _leading_edge(xs, foam, -1, 0.25, 4)
        o.edge_right = _leading_edge(xs, foam, 1, 0.25, 4)
        if foam.sum() > 0.5:
            o.centroid = float((foam * xs).sum() / foam.sum())
        obs_by_frame[i] = o
        if progress:
            progress(n, len(order))

    return {
        "obs": obs_by_frame,
        "W": W, "H": H, "scale": scale,
        "wall_side_guess": wall_side,
        "wall_start_x": wall_start_x,
        "tap_frame": tap_frame, "start_frame": start_frame,
    }
