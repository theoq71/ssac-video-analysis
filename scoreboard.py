"""Read the results board if the coach filmed it after the race.

Meet scoreboards show the event name, heat, and one row per lane with the
swimmer's name, team and time. Reading it gives the official time for the
lane (the most accurate finish possible) and the event's stroke and
distance, with no typing.
"""

from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

ROW_RE = re.compile(r"^\s*(\d)\s*([A-Z][A-Z'\-. ]+?,\s*[A-Z][A-Z'\-. ]*)\b.*?(\d{1,2}[.:,]\d{2}(?:[.,]\d{2})?)\s*(\d)?\s*$")
TIME_RE = re.compile(r"(\d{1,2})[:.](\d{2})[.,](\d{2})|(\d{2})[.,](\d{2})\b")
EVENT_RE = re.compile(r"(\d{2,4})\s*(?:M|METER|METRE|YARD|Y)\S*\s+(BUTTERFLY|FLY|FREESTYLE|FREE|BACKSTROKE|BACK|BREASTSTROKE|BREAST|IM|MEDLEY)", re.I)


def _candidate_frames(video, seconds: float = 8.0, count: int = 8) -> list[int]:
    n = video.frame_count
    start = max(int(n - seconds * video.fps), 0)
    step = max((n - start) // count, 1)
    return list(range(start, n, step))[:count]


def _text_from(rgb: np.ndarray) -> str:
    if pytesseract is None:
        return ""
    # LED text is coloured (red, yellow, white, blue) on a dark panel: the
    # brightest channel keeps every colour bright
    bright = rgb.max(axis=2)
    fx = min(3.0, max(1.0, 1500.0 / max(bright.shape[1], 1)))
    big = cv2.resize(bright, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (3, 3), 0)
    th = cv2.adaptiveThreshold(big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, -25)
    try:
        return pytesseract.image_to_string(th, config="--psm 6", timeout=15)
    except Exception:  # noqa: BLE001
        return ""


def _board_region(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Crop to the area dense with small, bright, saturated marks (LED text)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    led = ((v > 140) & (s > 110)).astype(np.uint8)
    # keep only small components (letters), not tents or shirts
    n, labels, stats, _ = cv2.connectedComponentsWithStats(led)
    small = np.zeros_like(led)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if 3 <= h <= 40 and 2 <= w <= 60 and area <= 800:
            small[labels == k] = 1
    dens = cv2.blur(small.astype(np.float32), (120, 60))
    H, W = dens.shape
    if dens.max() < 0.02:
        return None
    hot = (dens > max(0.35 * dens.max(), 0.02)).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(hot)
    if n2 < 2:
        return None
    areas = st2[1:, cv2.CC_STAT_AREA]
    big = [k for k in range(1, n2) if st2[k, cv2.CC_STAT_AREA] >= 0.2 * areas.max()]
    x0 = min(st2[k, cv2.CC_STAT_LEFT] for k in big)
    y0 = min(st2[k, cv2.CC_STAT_TOP] for k in big)
    x1 = max(st2[k, cv2.CC_STAT_LEFT] + st2[k, cv2.CC_STAT_WIDTH] for k in big)
    y1 = max(st2[k, cv2.CC_STAT_TOP] + st2[k, cv2.CC_STAT_HEIGHT] for k in big)
    if x1 - x0 < 0.12 * W or y1 - y0 < 0.08 * H:
        return None
    x0, y0 = max(x0 - 40, 0), max(y0 - 40, 0)
    x1, y1 = min(x1 + 40, W), min(y1 + 40, H)
    crop = rgb[y0:y1, x0:x1]
    # a results board is a dark panel; water and deck are not
    vv = v[y0:y1, x0:x1]
    if (vv < 80).mean() < 0.3:
        return None
    return crop


def _time_from_line(line: str) -> Optional[float]:
    """The result time at the end of a board row. Boards print the time and
    the place digit next to each other, so '26.72 4' often reads as 26724."""
    tail = line[-14:]
    tail = tail.replace("O", "0").replace("o", "0").replace("B", "8").replace("S", "5").replace("I", "1").replace("l", "1")
    m = re.search(r"(\d{1,2})[.:,](\d{2})[.,](\d{2})\s*\d?\s*$", tail)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100
    m = re.search(r"(\d{2})[.,](\d{2})\s*\d?\s*$", tail)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 100
    m = re.search(r"(\d{2})(\d{2})(\d)\s*$", tail)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 100
    m = re.search(r"(\d{2})(\d{2})\s*$", tail)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 100
    return None


def parse_board_text(text: str) -> dict:
    out = {"event": None, "distance_m": None, "stroke": None, "rows": []}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = EVENT_RE.search(line.upper())
        if m and out["event"] is None:
            dist = int(m.group(1))
            stroke = m.group(2).upper()
            stroke = {"FLY": "Butterfly", "BUTTERFLY": "Butterfly", "FREE": "Freestyle", "FREESTYLE": "Freestyle",
                      "BACK": "Backstroke", "BACKSTROKE": "Backstroke", "BREAST": "Breaststroke",
                      "BREASTSTROKE": "Breaststroke", "IM": "IM", "MEDLEY": "IM"}.get(stroke, stroke.title())
            out["event"] = line
            out["distance_m"] = dist
            out["stroke"] = stroke
            continue
        # rows: a lane digit (or a look-alike letter), a NAME, and a time
        fixed = line.lstrip("|:.,;'\" _-")
        lead = {"S": "5", "s": "5", "I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "B": "8", "Z": "2", "G": "6", "T": "7"}
        if fixed and fixed[0] in lead and len(fixed) > 1 and fixed[1].isalpha():
            fixed = lead[fixed[0]] + fixed[1:]
        mm = re.match(r"^\s*([1-9])\s*([A-Za-z@][A-Za-z@'\-]+)\s*[,.]?\s*([A-Za-z][A-Za-z'\-]*)", fixed)
        secs = _time_from_line(fixed)
        if mm and secs is not None:
            last = mm.group(2).upper().replace("@", "A")
            out["rows"].append({"lane": int(mm.group(1)), "last": last, "first": mm.group(3).upper(),
                                "time_s": secs})
    return out


def read_scoreboard(video, seconds: float = 8.0) -> dict:
    """Look through the end of the video for a results board and read it.
    Rows seen in several frames are trusted more (their times must agree)."""
    votes: dict = {}
    events = []
    for i in _candidate_frames(video, seconds):
        rgb = video.frame(i)
        if rgb is None:
            continue
        region = _board_region(rgb)
        if region is None:
            continue
        parsed = parse_board_text(_text_from(region))
        if parsed["event"]:
            events.append(parsed)
        for r in parsed["rows"]:
            key = (r["lane"], r["last"])
            votes.setdefault(key, []).append(r)
    rows = []
    for (lane, last), rs in votes.items():
        times = [r["time_s"] for r in rs]
        med = float(np.median(times))
        agree = sum(abs(t - med) < 0.02 for t in times)
        if agree >= 1:
            rows.append({"lane": lane, "last": last, "first": rs[0]["first"], "time_s": med, "seen": len(rs)})
    rows.sort(key=lambda r: r["lane"])
    ev = max(events, key=lambda e: 1) if events else None
    return {"rows": rows, "event": ev["event"] if ev else None,
            "distance_m": ev["distance_m"] if ev else None, "stroke": ev["stroke"] if ev else None}


def match_row(board: dict, name: Optional[str] = None, lane: Optional[int] = None) -> Optional[dict]:
    rows = board.get("rows", [])
    if not rows:
        return None
    if name:
        parts = [p.strip().upper() for p in re.split(r"[ ,]+", name) if p.strip()]
        for r in rows:
            if any(p and (p in r["last"] or p in r["first"]) for p in parts):
                return r
    if lane is not None:
        for r in rows:
            if r["lane"] == lane:
                return r
    return None
