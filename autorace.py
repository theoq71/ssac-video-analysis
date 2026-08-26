"""End-to-end automatic analysis of one race video.

    result = analyze_race(video_path, tap=(x, y), swimmer_name=...)

The only human input is the tap on the swimmer in the start frame. Every
other number is derived from the footage. Each derived value carries a
confidence note so the results page can say how much to trust it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from engine import scan
from metrics import RaceInput, analyze
from resolve import resolve
from scoreboard import match_row, read_scoreboard
from video import VideoSource, detect_start_signal, make_proxy

POOL_PROFILES = {
    # distances from the start wall at which the lane-rope pattern changes
    "50 m (long course)": {"length": 50.0, "marks": [5.0, 15.0, 25.0, 35.0, 45.0]},
    "25 m (short course meters)": {"length": 25.0, "marks": [5.0, 15.0]},
    "25 yd (short course yards)": {"length": 22.86, "marks": [4.57, 13.72]},
}

FINISH_GLIDE_S = 0.7   # the wake settles a little before the hand touches (the final reach)


def start_frame_for(video_path: str, fps: float) -> dict:
    """Frame of the horn, or a sensible fallback."""
    g = detect_start_signal(video_path, fps)
    if g and g.get("frame") is not None:
        return {"frame": int(g["frame"]), "source": "horn", "confidence": g.get("confidence", 0.5)}
    return {"frame": 0, "source": "video start", "confidence": 0.0}


def stroke_rhythm(obs: dict, start: int, fps: float, t_from: float, t_to: float) -> dict:
    """Stroke rate over time from the pulsing of the white water.

    Each arm entry throws up a burst of spray, so the amount of white water
    in the lane rises and falls once per stroke. Autocorrelation in sliding
    windows finds that period.
    """
    frames = sorted(obs)
    fa = np.array([obs[i].foam_amount for i in frames], float)
    k = max(int(2 * fps) | 1, 3)
    trend = np.convolve(fa, np.ones(k) / k, mode="same")
    d = fa - trend
    series = []
    win = int(3.0 * fps)
    hop = int(0.5 * fps)
    f0 = start + int(t_from * fps)
    f1 = start + int(t_to * fps)
    idx = {f: n for n, f in enumerate(frames)}
    for f in range(f0, f1 - win, hop):
        if f not in idx:
            continue
        k0 = idx[f]
        seg = d[k0:k0 + win]
        if len(seg) < win:
            break
        seg = seg - seg.mean()
        if np.abs(seg).sum() < 1e-6:
            continue
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        ac = ac / (ac[0] + 1e-9)
        best = None
        for L in range(int(0.45 * fps), int(2.2 * fps)):
            if L + 1 < len(ac) and ac[L] > ac[L - 1] and ac[L] >= ac[L + 1] and (best is None or ac[L] > best[1]):
                best = (L, ac[L])
        if best and best[1] > 0.15:
            series.append({"t_s": (f + win / 2 - start) / fps, "period_s": best[0] / fps,
                           "rate_per_min": 60.0 * fps / best[0], "strength": float(best[1])})
    return {"series": series}


def synth_strokes(rhythm: dict, start: int, fps: float, t_from: float, t_to: float) -> list[int]:
    """Stroke times consistent with the measured rate (phase is unknown, so
    the first stroke is placed half a period after the breakout)."""
    s = rhythm["series"]
    if not s:
        return []
    ts = np.array([r["t_s"] for r in s])
    per = np.array([r["period_s"] for r in s])
    out = []
    t = t_from + 0.5 * float(np.interp(t_from, ts, per))
    while t < t_to:
        out.append(start + int(round(t * fps)))
        t += float(np.interp(t, ts, per))
    return out


def analyze_race(video_path: str, tap: tuple[float, float], workdir: str, swimmer_name: str = "",
                 pool: str = "50 m (long course)", progress=None) -> dict:
    """Run everything. `tap` is (x, y) in the coordinates of the start frame
    image handed to the coach (full source resolution)."""
    def say(stage, frac):
        if progress:
            progress(stage, frac)

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    proxy = work / "proxy.mp4"
    if not proxy.exists():
        say("Preparing the video", 0.0)
        ok, msg = make_proxy(video_path, str(proxy), progress=lambda f: say("Preparing the video", f), width=1280)
        if not ok:
            raise RuntimeError(msg)
    src = VideoSource(video_path)
    fps = src.fps
    src.release()
    video = VideoSource(str(proxy))

    say("Listening for the horn", 0.0)
    st = start_frame_for(video_path, fps)
    start = st["frame"]

    say("Following the swimmer", 0.0)
    # the tap was made on the full-resolution start frame; scan expects that
    scan_res = scan(video, tap=tap, tap_frame=start, start_frame=start,
                    progress=lambda n, N: say("Following the swimmer", n / max(N, 1)))
    profile = POOL_PROFILES.get(pool, POOL_PROFILES["50 m (long course)"])
    say("Working out the splits", 0.0)
    res = resolve(scan_res, fps=fps, course_length=profile["length"], rope_marks=profile["marks"],
                  finish_glide_s=FINISH_GLIDE_S)

    events = dict(res["events"])
    notes = []
    confidence = {}
    finish_source = "video"
    if res["finish_frame"] is not None:
        events["finish"] = int(round(res["finish_frame"] + FINISH_GLIDE_S * fps))
        confidence["finish"] = "estimated from the video (about 0.4 s either way); type the official time to replace it"

    say("Reading the results board", 0.0)
    board = {}
    try:
        board = read_scoreboard(video)
    except Exception:  # noqa: BLE001
        board = {}
    row = match_row(board, name=swimmer_name) if board else None
    if row and res["finish_frame"] is not None:
        t_video = (events["finish"] - start) / fps
        if abs(row["time_s"] - t_video) < 1.5:
            events["finish"] = start + int(round(row["time_s"] * fps))
            finish_source = "results board"
            confidence["finish"] = f"official time {row['time_s']:.2f} read from the results board"
        else:
            notes.append(f"The results board shows {row['time_s']:.2f} for {row['last'].title()}, which does not match the video; using the video.")

    # breakout: the white water reappears after the underwater phase
    frames = sorted(scan_res["obs"])
    entry = events.get("entry")
    if entry is not None:
        quiet = False
        for i in frames:
            if i <= entry + int(0.6 * fps):
                continue
            fa = scan_res["obs"][i].foam_amount
            if fa < 0.02:
                quiet = True
            elif quiet and fa > 0.06:
                later = [scan_res["obs"][j].foam_amount for j in frames if i <= j < i + int(0.5 * fps)]
                if later and np.mean(later) > 0.05:
                    events["breakout"] = i
                    break

    # strokes from the rhythm of the spray, between breakout and finish
    t_bo = ((events.get("breakout") or events.get("entry") or start) - start) / fps
    t_fin = ((events.get("finish") or frames[-1]) - start) / fps
    rhythm = stroke_rhythm(scan_res["obs"], start, fps, max(t_bo, 0.5), t_fin)
    strokes = synth_strokes(rhythm, start, fps, t_bo + 0.2, t_fin - 0.2)

    stroke_type = board.get("stroke") if board else None
    stroke_unit = "stroke"
    for m in res["estimated"]:
        confidence[f"m{int(m)}"] = "no mark on the ropes here; interpolated between the neighbouring splits"
    for m in res["crossings"]:
        if m not in res["estimated"] and m < profile["length"]:
            confidence[f"m{int(m)}"] = "the swimmer's white water reaching the mark on the lane ropes"
    if events.get("entry") is not None:
        confidence["entry"] = "first splash in the lane after the horn"
    confidence["start"] = st["source"]

    inp = RaceInput(fps=fps, course_length=profile["length"],
                    events={k: v for k, v in events.items()},
                    strokes=strokes, entry_distance_m=res["entry_distance_m"], stroke_unit=stroke_unit)
    metrics = analyze(inp)
    metrics["rhythm"] = rhythm["series"]
    if rhythm["series"]:
        rates = [r["rate_per_min"] for r in rhythm["series"]]
        metrics["summary"]["stroke_rate_per_min"] = float(np.median(rates))
        metrics["summary"]["stroke_count"] = len(strokes)
        confidence["strokes"] = "rate measured from the rhythm of the spray; count is rate times swimming time"

    from annotate import per_second_speeds, phase_summary
    out = {
        "video": str(video_path),
        "fps": fps,
        "start": st,
        "events": events,
        "crossings": {str(k): v for k, v in res["crossings"].items()},
        "estimated": sorted(res["estimated"]),
        "entry_distance_m": res["entry_distance_m"],
        "finish_source": finish_source,
        "board": {"event": board.get("event"), "stroke": board.get("stroke"), "distance_m": board.get("distance_m"),
                  "rows": board.get("rows", [])} if board else None,
        "stroke_type": stroke_type,
        "confidence": confidence,
        "notes": notes + metrics.get("notes", []),
        "metrics": metrics,
        "direction": res["direction"],
        "work_width": scan_res["W"],
    }
    out["phases"] = phase_summary(out)
    out["per_second"] = per_second_speeds(out)
    (work / "result.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def review_frames(video_path: str, workdir: str, result: dict, size: int = 640) -> dict:
    """Small pictures of the key moments for the results page."""
    import cv2
    work = Path(workdir)
    video = VideoSource(str(work / "proxy.mp4"))
    fps = result["fps"]
    start = result["events"]["start"]
    out = {}
    labels = {"start": "Start", "entry": "Dive entry", "breakout": "Breakout", "m15": "15 m", "m25": "25 m",
              "m35": "35 m", "m45": "45 m", "finish": "Finish"}
    for key, label in labels.items():
        f = result["events"].get(key)
        if f is None:
            continue
        rgb = video.frame(int(f))
        if rgb is None:
            continue
        img = cv2.resize(rgb, (size, int(size * rgb.shape[0] / rgb.shape[1])))
        t = (f - start) / fps
        cv2.rectangle(img, (0, 0), (size, 28), (0, 0, 0), -1)
        cv2.putText(img, f"{label}  {t:.2f} s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        path = work / f"review_{key}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        out[key] = path.name
    return out
