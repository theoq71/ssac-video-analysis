"""Metrics engine for the SSAC swimming video analyzer.

Everything in here is pure computation on marked frame numbers and pixel
coordinates. The Streamlit app collects the marks; this module turns them into
times, distances, stroke rates and speeds.

Conventions
-----------
* Frames are integer indices into the video (0-based). Times are seconds from
  the start signal frame.
* Distances are meters from the start wall along the lane.
* A "stroke" is whatever the user tapped consistently (one arm entry, or one
  full cycle). Rates are reported per minute of the tapped unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Distance calibration along the lane (handles perspective from a side camera)
# ---------------------------------------------------------------------------

@dataclass
class RefPoint:
    x: float
    y: float
    meters: float
    label: str = ""


@dataclass
class LaneCalibration:
    """Maps pixel positions in one frame to meters along the lane.

    The user clicks landmarks on the lane whose distance from the start wall is
    known (wall = 0 m, backstroke flags = 5 m, 15 m marker, and so on). The
    landmarks are projected onto their best-fit line, and the 1-D coordinate
    along that line is mapped to meters with a projective (perspective) fit
    when three or more landmarks are available, or a linear fit with two.
    """

    refs: list[RefPoint] = field(default_factory=list)

    def __post_init__(self):
        self._origin = None
        self._dir = None
        self._coef = None
        self._mode = None
        if len(self.refs) >= 2:
            self.fit()

    # -- fitting -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._coef is not None

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    def fit(self):
        pts = np.array([[r.x, r.y] for r in self.refs], dtype=float)
        d = np.array([r.meters for r in self.refs], dtype=float)
        if len(pts) < 2:
            raise ValueError("Need at least two reference points")
        self._origin = pts.mean(axis=0)
        centered = pts - self._origin
        if len(pts) == 2:
            v = centered[1] - centered[0]
        else:
            # principal direction of the landmarks
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            v = vt[0]
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Reference points coincide")
        self._dir = v / norm
        s = centered @ self._dir

        if len(pts) >= 3 and len(np.unique(np.round(s, 6))) >= 3:
            # d = (a*s + b) / (c*s + 1)  ->  a*s + b - c*(d*s) = d
            A = np.column_stack([s, np.ones_like(s), -d * s])
            coef, *_ = np.linalg.lstsq(A, d, rcond=None)
            a, b, c = coef
            # reject a degenerate fit (pole inside the range of the landmarks)
            denom = c * s + 1
            if np.all(denom > 1e-9) or np.all(denom < -1e-9):
                self._coef = (a, b, c)
                self._mode = "projective"
                return
        A = np.column_stack([s, np.ones_like(s)])
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        self._coef = (coef[0], coef[1], 0.0)
        self._mode = "linear"

    # -- evaluation --------------------------------------------------------
    def _s(self, x: float, y: float) -> float:
        p = np.array([x, y], dtype=float) - self._origin
        return float(p @ self._dir)

    def distance_at(self, x: float, y: float) -> float:
        if not self.ready:
            raise ValueError("Calibration is not fitted")
        a, b, c = self._coef
        s = self._s(x, y)
        return float((a * s + b) / (c * s + 1.0))

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Closest point on the lane line, in pixels (for drawing)."""
        s = self._s(x, y)
        p = self._origin + s * self._dir
        return float(p[0]), float(p[1])

    def residuals(self) -> list[float]:
        if not self.ready:
            return []
        return [self.distance_at(r.x, r.y) - r.meters for r in self.refs]

    def to_dict(self) -> dict:
        return {"refs": [r.__dict__ for r in self.refs]}

    @classmethod
    def from_dict(cls, d: dict) -> "LaneCalibration":
        return cls(refs=[RefPoint(**r) for r in d.get("refs", [])])


# ---------------------------------------------------------------------------
# Monotone cubic interpolation (Fritsch-Carlson) for the distance-time curve
# ---------------------------------------------------------------------------

class MonotoneCubic:
    """Shape-preserving interpolant through (t, d) knots with derivative."""

    def __init__(self, t, d):
        t = np.asarray(t, dtype=float)
        d = np.asarray(d, dtype=float)
        order = np.argsort(t)
        t, d = t[order], d[order]
        keep = np.concatenate([[True], np.diff(t) > 1e-9])
        self.t, self.d = t[keep], d[keep]
        n = len(self.t)
        if n < 2:
            raise ValueError("Need at least two knots")
        h = np.diff(self.t)
        delta = np.diff(self.d) / h
        m = np.zeros(n)
        if n == 2:
            m[:] = delta[0]
        else:
            m[0] = delta[0]
            m[-1] = delta[-1]
            for i in range(1, n - 1):
                if delta[i - 1] * delta[i] <= 0:
                    m[i] = 0.0
                else:
                    w1 = 2 * h[i] + h[i - 1]
                    w2 = h[i] + 2 * h[i - 1]
                    m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        self.m = m
        self.h = h
        self.delta = delta

    def _locate(self, x):
        idx = np.searchsorted(self.t, x, side="right") - 1
        return np.clip(idx, 0, len(self.t) - 2)

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        i = self._locate(x)
        h = self.h[i]
        s = (x - self.t[i]) / h
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * self.d[i] + h10 * h * self.m[i] + h01 * self.d[i + 1] + h11 * h * self.m[i + 1]

    def derivative(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        i = self._locate(x)
        h = self.h[i]
        s = (x - self.t[i]) / h
        dh00 = 6 * s**2 - 6 * s
        dh10 = 3 * s**2 - 4 * s + 1
        dh01 = -6 * s**2 + 6 * s
        dh11 = 3 * s**2 - 2 * s
        return (dh00 * self.d[i] + dh10 * h * self.m[i] + dh01 * self.d[i + 1] + dh11 * h * self.m[i + 1]) / h


# ---------------------------------------------------------------------------
# Race analysis
# ---------------------------------------------------------------------------

EVENT_ORDER = ["start", "entry", "breakout", "m15", "m25", "m35", "finish"]

EVENT_LABELS = {
    "start": "Start signal",
    "entry": "Dive entry (hands hit water)",
    "breakout": "Breakout (head surfaces)",
    "m15": "Head passes 15 m",
    "m25": "Head passes 25 m",
    "m35": "Head passes 35 m",
    "finish": "Wall touch",
}


def split_marks_for_course(course_length: float) -> list[float]:
    """Intermediate split distances that apply to this course length."""
    return [m for m in (15.0, 25.0, 35.0) if m < course_length - 0.5]


def event_key_for_distance(meters: float) -> str:
    return f"m{int(round(meters))}"


@dataclass
class RaceInput:
    fps: float
    course_length: float = 50.0
    events: dict = field(default_factory=dict)      # event key -> frame index
    strokes: list = field(default_factory=list)     # frame indices
    entry_distance_m: Optional[float] = None
    stroke_unit: str = "stroke"                     # label only


def frame_to_time(frame: Optional[int], start_frame: int, fps: float) -> Optional[float]:
    if frame is None:
        return None
    return (frame - start_frame) / fps


def analyze(inp: RaceInput) -> dict:
    """Compute every metric that the marked data supports.

    Returns a dict with keys: ok, missing, times, splits, segments, strokes,
    stroke_rate_series, speed_curve, summary, notes.
    """
    out = {"ok": False, "missing": [], "notes": []}
    ev = inp.events or {}
    fps = float(inp.fps)
    start = ev.get("start")
    if start is None:
        out["missing"].append("start")
        return out

    T = lambda k: frame_to_time(ev.get(k), start, fps)  # noqa: E731

    # ---- split times ----------------------------------------------------
    marks = split_marks_for_course(inp.course_length)
    times = {"entry": T("entry"), "breakout": T("breakout"), "finish": T("finish")}
    for m in marks:
        times[event_key_for_distance(m)] = T(event_key_for_distance(m))
    out["times"] = times

    # distance-time knots that are actually marked
    knots = [(0.0, 0.0)]
    entry_d = inp.entry_distance_m
    if times["entry"] is not None and entry_d is not None and times["entry"] > 0:
        knots.append((times["entry"], float(entry_d)))
    for m in marks:
        t = times[event_key_for_distance(m)]
        if t is not None:
            knots.append((t, m))
    if times["finish"] is not None:
        knots.append((times["finish"], float(inp.course_length)))
    knots.sort()
    # drop any knot that is not strictly increasing in time and distance
    clean = [knots[0]]
    for t, d in knots[1:]:
        if t > clean[-1][0] and d > clean[-1][1]:
            clean.append((t, d))
        else:
            out["notes"].append(f"Ignored out-of-order mark at {d:g} m / {t:.2f} s")
    knots = clean

    curve = MonotoneCubic([k[0] for k in knots], [k[1] for k in knots]) if len(knots) >= 2 else None

    # ---- splits table ---------------------------------------------------
    splits = []
    prev_t, prev_d = 0.0, 0.0
    for t, d in knots[1:]:
        if d == entry_d and t == times["entry"]:
            continue
        splits.append({
            "distance_m": d,
            "time_s": t,
            "segment": f"{prev_d:g}-{d:g} m",
            "segment_time_s": t - prev_t,
            "segment_speed_mps": (d - prev_d) / (t - prev_t),
        })
        prev_t, prev_d = t, d
    out["splits"] = splits

    # ---- breakout distance estimate --------------------------------------
    breakout_d = None
    if times["breakout"] is not None and curve is not None:
        breakout_d = float(curve(times["breakout"])[0])
        out["notes"].append("Breakout distance is interpolated from the marked splits, not measured")

    # ---- strokes ----------------------------------------------------------
    stroke_times = sorted(frame_to_time(f, start, fps) for f in (inp.strokes or []))
    stroke_times = [t for t in stroke_times if t is not None and t >= 0]
    out["strokes"] = {"count": len(stroke_times), "times_s": stroke_times, "unit": inp.stroke_unit}

    rate_series = []
    if len(stroke_times) >= 2:
        st_arr = np.array(stroke_times)
        cycle = np.diff(st_arr)
        mids = (st_arr[:-1] + st_arr[1:]) / 2
        for i in range(len(cycle)):
            rate_series.append({
                "t_s": float(mids[i]),
                "cycle_time_s": float(cycle[i]),
                "rate_per_min": float(60.0 / cycle[i]) if cycle[i] > 0 else None,
            })
    out["stroke_rate_series"] = rate_series

    # cumulative stroke count as a function of time (piecewise linear), so
    # segments can contain fractional strokes instead of being quantized
    def strokes_between(t0: float, t1: float) -> Optional[float]:
        if len(stroke_times) < 2:
            return None
        st_arr = np.array(stroke_times)
        idx = np.arange(len(st_arr), dtype=float)
        n0 = float(np.interp(t0, st_arr, idx, left=0.0, right=idx[-1]))
        n1 = float(np.interp(t1, st_arr, idx, left=0.0, right=idx[-1]))
        return max(n1 - n0, 0.0)

    def stroking_time_between(t0: float, t1: float) -> float:
        """Portion of [t0, t1] that lies inside the tapped stroke sequence."""
        if len(stroke_times) < 2:
            return 0.0
        lo, hi = max(t0, stroke_times[0]), min(t1, stroke_times[-1])
        return max(hi - lo, 0.0)

    # ---- per-segment stroke metrics --------------------------------------
    # Stroke segments start where swimming starts (breakout if marked,
    # otherwise the first tapped stroke), then follow the split marks.
    segments = []
    seg_bounds = []
    swim_start_t = times["breakout"] if times["breakout"] is not None else (stroke_times[0] if stroke_times else None)
    if swim_start_t is not None and curve is not None:
        swim_start_d = float(curve(swim_start_t)[0])
        bounds = [(swim_start_t, swim_start_d)]
        for t, d in knots[1:]:
            if t > swim_start_t and (d != entry_d or t != times["entry"]):
                bounds.append((t, d))
        for (t0, d0), (t1, d1) in zip(bounds[:-1], bounds[1:]):
            n = strokes_between(t0, t1)
            dist = d1 - d0
            dur = t1 - t0
            speed = dist / dur if dur > 0 else None
            dps = dist / n if n and n > 0 else None
            stroking = stroking_time_between(t0, t1)
            rate = n / stroking * 60 if n and stroking > 0 else None
            si = speed * dps if speed is not None and dps is not None else None
            segments.append({
                "segment": f"{d0:.1f}-{d1:.1f} m",
                "start_m": d0, "end_m": d1,
                "distance_m": dist,
                "time_s": dur,
                "speed_mps": speed,
                "strokes": n,
                "stroke_rate_per_min": rate,
                "distance_per_stroke_m": dps,
                "stroke_index": si,
            })
            seg_bounds.append((t0, t1, dps))
    out["segments"] = segments

    # ---- per-stroke speed (segment DPS / cycle time) ---------------------
    for r in rate_series:
        dps = None
        for t0, t1, seg_dps in seg_bounds:
            if t0 <= r["t_s"] <= t1:
                dps = seg_dps
                break
        r["speed_mps"] = (dps / r["cycle_time_s"]) if dps and r["cycle_time_s"] > 0 else None
        if curve is not None:
            r["curve_speed_mps"] = float(curve.derivative(r["t_s"])[0])

    # ---- continuous speed curve from the split marks ----------------------
    speed_curve = []
    if curve is not None and len(knots) >= 3:
        ts = np.arange(0.0, knots[-1][0] + 1e-9, 0.1)
        ds = curve(ts)
        vs = curve.derivative(ts)
        speed_curve = [{"t_s": float(t), "distance_m": float(d), "speed_mps": float(v)} for t, d, v in zip(ts, ds, vs)]
    elif curve is not None:
        out["notes"].append("Speed curve needs at least two marked distances after the start")
    out["speed_curve"] = speed_curve

    # ---- whole-lap summary --------------------------------------------------
    summary = {
        "final_time_s": times["finish"],
        "entry_distance_m": entry_d,
        "entry_time_s": times["entry"],
        "breakout_time_s": times["breakout"],
        "breakout_distance_m": breakout_d,
        "stroke_count": len(stroke_times),
    }
    for m in marks:
        summary[f"time_to_{int(m)}m_s"] = times[event_key_for_distance(m)]
    if times["finish"] is not None:
        summary["average_speed_mps"] = inp.course_length / times["finish"]
    if segments:
        total_d = sum(s["distance_m"] for s in segments)
        total_t = sum(s["time_s"] for s in segments)
        total_n = sum(s["strokes"] for s in segments if s["strokes"] is not None)
        if total_t > 0 and total_n > 0:
            v = total_d / total_t
            dps = total_d / total_n
            summary["swim_speed_mps"] = v
            summary["distance_per_stroke_m"] = dps
            summary["stroke_index"] = v * dps
            summary["swim_distance_m"] = total_d
    if len(stroke_times) >= 2:
        span = stroke_times[-1] - stroke_times[0]
        if span > 0:
            summary["stroke_rate_per_min"] = (len(stroke_times) - 1) / span * 60
            summary["mean_cycle_time_s"] = span / (len(stroke_times) - 1)
    out["summary"] = summary

    required = ["finish"] + [event_key_for_distance(m) for m in marks]
    out["missing"] = [k for k in required if ev.get(k) is None]
    out["ok"] = True
    return out


def fmt_time(t: Optional[float]) -> str:
    if t is None:
        return "-"
    if t >= 60:
        m = int(t // 60)
        return f"{m}:{t - 60 * m:05.2f}"
    return f"{t:.2f}"
