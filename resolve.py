"""Stage 2 of the automatic analysis: from per-frame observations to race facts.

Landmarks (the walls and the colour-change marks on the ropes) are tracked in
image coordinates only over the seconds in which they are visible, and
carried a little further with the camera motion. Splits come from the
swimmer's white water crossing those landmarks; distances with no physical
mark (25 m) are interpolated between the neighbouring ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from metrics import LaneCalibration, RefPoint


@dataclass
class Landmark:
    kind: str                                 # "mark", "wall_start", "wall_far"
    obs: list = field(default_factory=list)   # [(frame, x)] where it was seen
    x_by_frame: dict = field(default_factory=dict)   # frame -> x (seen or dead-reckoned)
    meters: Optional[float] = None
    cross_frame: Optional[float] = None

    @property
    def first(self):
        return self.obs[0][0]

    @property
    def last(self):
        return self.obs[-1][0]


def _median_filter(values: dict, frames: list, win: int = 9) -> dict:
    arr = np.array([np.nan if values.get(i) is None else values[i] for i in frames], float)
    out = {}
    h = win // 2
    for k, i in enumerate(frames):
        seg = arr[max(k - h, 0):k + h + 1]
        seg = seg[~np.isnan(seg)]
        out[i] = float(np.median(seg)) if seg.size >= 3 else None
    return out


def front_profile(o, direction: int, x_lo: float, x_hi: float, min_score: float = 0.4, min_run: int = 6) -> Optional[float]:
    """Leading edge of the swimmer's own white water: the lane's middle third
    must be white too, so a neighbour's splash spilling over a rope is ignored."""
    xs, f = o.foam_xs, o.foam
    core = getattr(o, "core", None)
    if xs is None:
        return None
    on = (f >= min_score) & (xs >= x_lo) & (xs <= x_hi)
    if core is not None:
        on &= core >= 0.35
    idx = range(len(on)) if direction < 0 else range(len(on) - 1, -1, -1)
    run = 0
    start = None
    for k in idx:
        if on[k]:
            if run == 0:
                start = k
            run += 1
            if run >= min_run:
                return float(xs[start])
        else:
            run = 0
    return None


def foam_centroid(o, x_lo: float, x_hi: float) -> Optional[float]:
    xs, f = o.foam_xs, o.foam
    if xs is None:
        return None
    m = (xs >= x_lo) & (xs <= x_hi)
    w = f[m]
    if w.sum() < 0.5:
        return None
    return float((w * xs[m]).sum() / w.sum())


def dead_reckon(S_by: dict, frames: list, seen: list, span: int, W: int) -> dict:
    """Carry a landmark's image x through the frames around its sightings
    using the camera pan: x(f) = x(nearest sighting) + pan since then."""
    out = {f: x for f, x in seen}
    if not seen:
        return out
    sf = np.array([f for f, _ in seen])
    sx = np.array([x for _, x in seen])
    lo, hi = seen[0][0] - span, seen[-1][0] + span
    for f in frames:
        if f < lo or f > hi or f in out:
            continue
        k = int(np.argmin(np.abs(sf - f)))
        x = sx[k] + (S_by[f] - S_by[int(sf[k])])
        if -0.5 * W < x < 1.5 * W:
            out[f] = float(x)
    return out


def resolve(scan_result: dict, fps: float, course_length: float = 50.0,
            rope_marks: Optional[list] = None, finish_glide_s: float = 0.0) -> dict:
    obs = scan_result["obs"]
    W = scan_result["W"]
    frames = sorted(obs)
    start = scan_result["start_frame"]
    rope_marks = rope_marks or [5.0, 15.0, 25.0, 35.0, 45.0]
    span = int(3.0 * fps)

    # ---- travel direction: which way the white water drifts, corrected for the pan
    fr, ps, S = [], [], 0.0
    S_by = {}
    for i in frames:
        S += obs[i].dx
        S_by[i] = S
        o = obs[i]
        if o.centroid is not None and o.foam_amount > 0.03:
            fr.append(i)
            ps.append(o.centroid - S)
    direction = -1
    if len(fr) >= 10:
        direction = 1 if np.polyfit(fr, ps, 1)[0] > 0 else -1

    # ---- start wall: the pool's end behind the swimmer, before the dive splash
    wall_start = Landmark(kind="wall_start")
    for i in frames[: int(1.2 * fps)]:
        o = obs[i]
        x = o.pool_x1 if direction < 0 else o.pool_x0
        if (direction < 0 and 0 < x < W - 6) or (direction > 0 and x > 6):
            wall_start.obs.append((i, float(x)))
    tap_wall = scan_result.get("wall_start_x")
    tap_frame = scan_result.get("tap_frame", start)
    if tap_wall is not None and tap_frame in S_by:
        # the deck edge found under the tapped swimmer is the best fix on the wall
        P0 = float(tap_wall - S_by[tap_frame])
        wall_start.obs = [(tap_frame, float(tap_wall))]
        wall_start.x_by_frame = {i: P0 + S_by[i] for i in frames[: int(6 * fps)]}
    elif wall_start.obs:
        # a single robust value carried by the camera motion
        xs_ = np.array([x - S_by[f] for f, x in wall_start.obs])
        P0 = float(np.median(xs_))
        wall_start.x_by_frame = {i: P0 + S_by[i] for i in frames[: int(6 * fps)]}

    # ---- far wall: the pool's end ahead of the swimmer, from its cleanest sightings
    wall_far = Landmark(kind="wall_far")
    cands = []
    for i in frames[int(len(frames) * 0.5):]:
        o = obs[i]
        x = o.pool_x0 if direction < 0 else o.pool_x1
        if (direction < 0 and x > 6) or (direction > 0 and 0 < x < W - 6):
            cands.append((i, float(x), x - S_by[i]))
    if len(cands) >= 5:
        Ps = np.array([c[2] for c in cands])
        # the wall is the furthest-forward cluster; splash fronts sit behind it
        srt = np.sort(Ps) if direction < 0 else -np.sort(-Ps)
        anchor = float(np.median(srt[: max(3, int(0.12 * srt.size))]))
        good = [c for c in cands if abs(c[2] - anchor) < 40]
        if len(good) >= 3:
            wall_far.obs = [(c[0], c[1]) for c in good]
            first_seen = good[0][0]
            Pw = float(np.median([c[2] for c in good]))
            wall_far.x_by_frame = {i: Pw + S_by[i] for i in frames if i >= first_seen - span}

    # ---- swimmer front and centroid, limited to the pool ---------------------------
    front = {}
    cent = {}
    for i in frames:
        o = obs[i]
        lo, hi = -1e9, 1e9
        xs_w = wall_start.x_by_frame.get(i)
        xf_w = wall_far.x_by_frame.get(i)
        if direction < 0:
            if xs_w is not None:
                hi = xs_w + 10
            if xf_w is not None:
                lo = xf_w - 10
        else:
            if xs_w is not None:
                lo = xs_w - 10
            if xf_w is not None:
                hi = xf_w + 10
        front[i] = front_profile(o, direction, lo, hi)
        cent[i] = foam_centroid(o, lo, hi)
    # a steadier front: the wake's centre of mass is smooth, and the front
    # sits a slowly varying distance ahead of it, so estimate that lead with
    # a wide rolling median and add it back (short outliers such as a
    # neighbour's splash drop out)
    front_raw = _median_filter(front, frames, 5)
    cP0 = {i: cent[i] - S_by[i] for i in frames if cent.get(i) is not None}
    cP0_s = _median_filter(cP0, frames, 9)
    lead = {}
    for i in frames:
        if front_raw.get(i) is not None and cP0_s.get(i) is not None:
            lead[i] = (front_raw[i] - S_by[i]) - cP0_s[i]
    lead_s = _median_filter(lead, frames, 91)
    front_s = {}
    for i in frames:
        if cP0_s.get(i) is not None and lead_s.get(i) is not None:
            front_s[i] = cP0_s[i] + lead_s[i] + S_by[i]
        else:
            front_s[i] = front_raw.get(i)

    # ---- entry: the splash of the hands striking the water ---------------------------
    # The flying body sweeps bright pixels along the lane before it lands, and
    # the water near the blocks may already be churned, so one bright frame is
    # not enough. The entry is the first burst of NEW white water near the
    # start wall that is still there half a second later: splash persists,
    # a body in flight does not.
    entry_frame = None
    entry_x = None
    idx = {f: n for n, f in enumerate(frames)}
    K = max(int(0.25 * fps), 2)

    def novel_foam(i):
        n = idx[i]
        if n < K:
            return None
        o, p = obs[i], obs[frames[n - K]]
        if o.foam_xs is None or p.foam_xs is None:
            return None
        shift = S_by[i] - S_by[frames[n - K]]
        prev = np.interp(o.foam_xs - shift, p.foam_xs, p.foam, left=0.0, right=0.0)
        return np.clip(o.foam - prev, 0.0, None)

    hold = int(0.4 * fps)
    for i in frames:
        if i < start + int(0.3 * fps):
            continue
        o = obs[i]
        xw = wall_start.x_by_frame.get(i)
        if xw is None:
            break
        nv = novel_foam(i)
        if nv is None:
            continue
        lo, hi = (xw - 420, xw + 10) if direction < 0 else (xw - 10, xw + 420)
        strong = (o.foam_xs >= lo) & (o.foam_xs <= hi) & (nv >= 0.3)
        if strong.sum() < 8:
            continue
        j = next((c for c in frames[idx[i]:] if c >= i + hold), None)
        if j is None:
            break
        oj = obs[j]
        if oj.foam_xs is None:
            continue
        drift = S_by[j] - S_by[i]
        later = np.interp(o.foam_xs[strong] + drift, oj.foam_xs, oj.foam, left=0.0, right=0.0)
        if (later >= 0.25).sum() >= 0.5 * strong.sum():
            entry_frame = i
            w = nv[strong]
            entry_x = float((w * o.foam_xs[strong]).sum() / w.sum())
            break

    # ---- finish: the swimmer stops at the far wall -------------------------------
    # The white water's front reaches the wall a little before the touch (the
    # bow wave arrives first); the wake's centre of mass keeps moving until
    # the hand hits, then stops. The finish is where that motion dies.
    finish_frame = None
    cP = {i: (cent[i] - S_by[i]) for i in frames if cent.get(i) is not None and obs[i].foam_amount > 0.04}
    cP_s = _median_filter(cP, frames, 9)
    reach = None
    if wall_far.x_by_frame:
        seq = [i for i in frames if i in wall_far.x_by_frame and i > start + 5 * fps]
        for k, i in enumerate(seq):
            e = front_s.get(i)
            if e is None:
                continue
            if (e - wall_far.x_by_frame[i]) * direction >= -20:
                later = [(front_s.get(j), j) for j in seq[k:k + 4]]
                if all(x is not None and (x - wall_far.x_by_frame[j]) * direction >= -28 for x, j in later):
                    reach = i
                    break
    valid = [i for i in frames if cP_s.get(i) is not None and i > start + 5 * fps]
    # tracking is trustworthy while the far wall is in the picture and the
    # wake is still visible; after the touch the camera usually swings away
    if wall_far.x_by_frame:
        valid = [i for i in valid if i in wall_far.x_by_frame and -20 <= wall_far.x_by_frame[i] <= W + 20
                 and obs[i].foam_amount > 0.08]
    # ... and until the camera swings away (a sudden fast pan)
    cut = None
    run = 0
    for i in valid:
        if reach is not None and i < reach:
            continue
        run = run + 1 if abs(obs[i].dx) > 12 else 0
        if run >= 3:
            cut = i - 3
            break
    if cut is not None:
        valid = [i for i in valid if i <= cut]
    if len(valid) > int(2 * fps):
        tail = [i for i in valid if i >= valid[-1] - int(0.8 * fps)]
        L = float(np.median([cP_s[i] for i in tail]))
        # the wake settles at L; the swimmer has all but stopped when it
        # comes within a hand's width of that level and stays there
        for k, i in enumerate(valid):
            if reach is not None and i < reach - int(0.3 * fps):
                continue
            if (cP_s[i] - L) * direction >= -15:
                later = [cP_s[j] for j in valid[k:k + int(0.3 * fps)]]
                if later and all((x - L) * direction >= -22 for x in later):
                    finish_frame = i
                    break
        if finish_frame is None and reach is not None:
            finish_frame = reach + int(0.8 * fps)

    # ---- marks: tracks by image-x continuity ------------------------------------------
    marks: list[Landmark] = []
    for i in frames:
        for x in obs[i].marks:
            best = None
            for t in marks:
                if i - t.last > int(1.0 * fps):
                    continue
                # predicted position: last sighting moved by the pan since
                pred = t.obs[-1][1] + (S_by[i] - S_by[t.last])
                d = abs(pred - x)
                if d < 35 and (best is None or d < best[0]):
                    best = (d, t)
            if best:
                best[1].obs.append((i, float(x)))
            else:
                marks.append(Landmark(kind="mark", obs=[(i, float(x))]))
    marks = [t for t in marks if len(t.obs) >= 8]
    # merge tracks that are the same mark seen again after a gap
    merged = []
    for t in sorted(marks, key=lambda t: t.first):
        hit = None
        for u in merged:
            if t.first > u.last and t.first - u.last < int(3.0 * fps):
                pred = u.obs[-1][1] + (S_by[t.first] - S_by[u.last])
                if abs(pred - t.obs[0][1]) < 60:
                    hit = u
                    break
        if hit:
            hit.obs.extend(t.obs)
        else:
            merged.append(t)
    marks = merged
    for t in marks:
        t.x_by_frame = dead_reckon(S_by, frames, t.obs, span, W)

    # ---- crossings -----------------------------------------------------------------
    def crossing(xmap: dict) -> Optional[float]:
        seq = [i for i in frames if i in xmap]
        prev = None
        for k, i in enumerate(seq):
            e = front_s.get(i)
            if e is None:
                continue
            rel = (e - xmap[i]) * direction
            if rel >= 0:
                later = [(front_s.get(j), j) for j in seq[k:k + 4]]
                if all(x is not None and (x - xmap[j]) * direction >= 0 for x, j in later):
                    if prev is not None and prev[1] < 0:
                        f0, r0 = prev
                        return f0 + (i - f0) * (-r0 / (rel - r0))
                    return float(i)
            prev = (i, rel)
        return None

    crossings = {}
    estimated = set()
    for t in marks:
        cf = crossing(t.x_by_frame)
        if cf is not None and not (t.first - 1.0 * fps <= cf <= t.last + 3.0 * fps):
            cf = None
        t.cross_frame = cf
    # identity: the mark seen together with the start wall is the first in
    # the profile; the rest are assigned in order so that the swimming speed
    # between consecutive crossings stays plausible
    T = None if finish_frame is None else (finish_frame - start) / fps
    v_avg = (course_length / T) if T else 1.7
    timed = [t for t in sorted(marks, key=lambda t: t.first) if t.cross_frame is not None]
    for t in marks:
        t.meters = None
    if timed:
        first = timed[0]
        rest = timed[1:]
        first.meters = rope_marks[0]
        remaining_marks = [m for m in rope_marks[1:] if m < course_length]
        import itertools
        best = None
        for n_use in range(len(rest), -1, -1):
            for chosen in itertools.combinations(range(len(rest)), n_use):
                for dists in itertools.combinations(remaining_marks, n_use):
                    pts = [(first.cross_frame, first.meters)] + [(rest[c].cross_frame, d) for c, d in zip(chosen, dists)]
                    if T:
                        pts.append((finish_frame + finish_glide_s * fps, course_length))
                    ok = True
                    score = 0.0
                    for (f0, d0), (f1, d1) in zip(pts[:-1], pts[1:]):
                        dt = (f1 - f0) / fps
                        if dt <= 0:
                            ok = False
                            break
                        v = (d1 - d0) / dt
                        if not (0.55 * v_avg <= v <= 1.7 * v_avg):
                            ok = False
                            break
                        score += abs(np.log(v / v_avg))
                    if ok and (best is None or (n_use, -score) > (best[0], -best[1])):
                        best = (n_use, score, chosen, dists)
            if best is not None:
                break
        if best is not None:
            for c, d in zip(best[2], best[3]):
                rest[c].meters = d
    for t in marks:
        if t.meters is not None and t.cross_frame is not None:
            crossings[t.meters] = t.cross_frame
    if finish_frame is not None:
        # the wake settles a moment before the hand reaches the wall, so the
        # touch itself, not the settling, anchors the interpolated splits
        crossings[course_length] = float(finish_frame + finish_glide_s * fps)
    # sanity: crossings must come in distance order; drop offenders
    keys = sorted(k for k in crossings)
    last_f = start
    for k in keys:
        if crossings[k] <= last_f:
            crossings.pop(k)
        else:
            last_f = crossings[k]
    # distances without a mark: interpolate between the neighbours
    for m in (15.0, 25.0, 35.0):
        if m in crossings or m >= course_length:
            continue
        below = [k for k in crossings if k < m]
        above = [k for k in crossings if k > m]
        if below and above:
            k0, k1 = max(below), min(above)
            f0, f1 = crossings[k0], crossings[k1]
            crossings[m] = f0 + (f1 - f0) * (m - k0) / (k1 - k0)
            estimated.add(m)

    # ---- entry distance ------------------------------------------------------------
    entry_m = None
    if entry_frame is not None and entry_x is not None and entry_frame in wall_start.x_by_frame:
        pts = [(wall_start.x_by_frame[entry_frame], 0.0)]
        for t in marks:
            if t.meters is not None and entry_frame in t.x_by_frame and t.first <= entry_frame + span:
                pts.append((t.x_by_frame[entry_frame], t.meters))
        if len(pts) >= 2:
            cal = LaneCalibration([RefPoint(x=p, y=0.0, meters=m) for p, m in sorted(pts)])
            entry_m = float(cal.distance_at(entry_x, 0.0))

    events = {"start": start}
    if entry_frame is not None:
        events["entry"] = int(entry_frame)
    if finish_frame is not None:
        events["finish"] = int(round(finish_frame))
    for m, cf in crossings.items():
        if m < course_length:
            events[f"m{int(m)}"] = int(round(cf))

    return {
        "direction": direction,
        "S": S_by,
        "wall_start": wall_start,
        "wall_far": wall_far,
        "marks": marks,
        "front": front_s,
        "centroid": cent,
        "events": events,
        "crossings": crossings,
        "estimated": estimated,
        "entry_distance_m": entry_m,
        "entry_x": entry_x,
        "finish_frame": finish_frame,
    }
