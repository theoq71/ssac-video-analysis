"""Phases, per-second speeds, and the annotated speed video.

The engine gives a distance-along-the-lane track for the whole lap. From it:
- the lap in three phases: dive (horn to hands-in), underwater (hands-in to
  breakout), swim (breakout to the wall), each with time, distance and speed;
- the swimmer's speed at every second of the race;
- a copy of the video with the race clock, the current speed and the phase
  drawn in the top-left corner while the swimmer is moving.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from metrics import MonotoneCubic
from video import VideoSource, ffmpeg_binary


def build_curve(result: dict):
    """Distance-over-time interpolant from the analysed events."""
    fps = result["fps"]
    start = result["events"]["start"]
    s = result["metrics"]["summary"]
    knots = [(0.0, 0.0)]
    if s.get("entry_time_s") is not None and s.get("entry_distance_m"):
        knots.append((s["entry_time_s"], s["entry_distance_m"]))
    if s.get("breakout_time_s") is not None and s.get("breakout_distance_m"):
        knots.append((s["breakout_time_s"], s["breakout_distance_m"]))
    for key, m in (("time_to_15m_s", 15.0), ("time_to_25m_s", 25.0), ("time_to_35m_s", 35.0)):
        if s.get(key) is not None:
            knots.append((s[key], m))
    course = 50.0
    for sp in result["metrics"].get("splits", []):
        course = max(course, sp["distance_m"])
    if s.get("final_time_s") is not None:
        knots.append((s["final_time_s"], course))
    knots.sort()
    clean = [knots[0]]
    for t, d in knots[1:]:
        if t > clean[-1][0] + 0.05 and d > clean[-1][1]:
            clean.append((t, d))
    if len(clean) < 2:
        return None
    return MonotoneCubic([k[0] for k in clean], [k[1] for k in clean])


def phase_summary(result: dict) -> list[dict]:
    """The lap in three phases: dive, underwater, swim."""
    s = result["metrics"]["summary"]
    T = s.get("final_time_s")
    out = []
    t_entry = s.get("entry_time_s")
    d_entry = s.get("entry_distance_m")
    t_bo = s.get("breakout_time_s")
    d_bo = s.get("breakout_distance_m")
    course = 50.0
    for sp in result["metrics"].get("splits", []):
        course = max(course, sp["distance_m"])

    def phase(name, t0, t1, d0, d1, note=""):
        if t0 is None or t1 is None or t1 <= t0:
            return None
        dist = None if d0 is None or d1 is None else d1 - d0
        return {"phase": name, "from_s": t0, "to_s": t1, "time_s": t1 - t0,
                "distance_m": dist, "speed_mps": (dist / (t1 - t0)) if dist else None, "note": note}

    p = phase("Dive (horn to hands in)", 0.0, t_entry, 0.0, d_entry)
    if p:
        out.append(p)
    p = phase("Underwater (hands in to breakout)", t_entry, t_bo, d_entry, d_bo,
              "breakout distance is estimated")
    if p:
        out.append(p)
    p = phase("Swim (breakout to the wall)", t_bo if t_bo is not None else t_entry, T,
              d_bo if d_bo is not None else d_entry, course)
    if p:
        out.append(p)
    return out


def per_second_speeds(result: dict) -> list[dict]:
    curve = build_curve(result)
    if curve is None:
        return []
    s = result["metrics"]["summary"]
    T = s.get("final_time_s")
    if T is None:
        return []
    out = []
    for t in range(1, int(np.floor(T)) + 1):
        v = float(curve.derivative(float(t))[0])
        d = float(curve(float(t))[0])
        out.append({"t_s": t, "speed_mps": round(max(v, 0.0), 2), "distance_m": round(d, 1)})
    return out


def _phase_at(result: dict, t: float) -> str:
    s = result["metrics"]["summary"]
    t_entry = s.get("entry_time_s")
    t_bo = s.get("breakout_time_s")
    if t_entry is not None and t < t_entry:
        return "DIVE"
    if t_bo is not None and t < t_bo:
        return "UNDERWATER"
    return "SWIM"


def _card_font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf", "Helvetica.ttc"] if bold
             else ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"])
    dirs = ["/usr/share/fonts/truetype/dejavu/", "/System/Library/Fonts/Supplemental/",
            "/System/Library/Fonts/", "/Library/Fonts/", ""]
    for d in dirs:
        for n in names:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_results_card(workdir: str, result: dict, out_name: str = "results.png") -> Optional[str]:
    """One shareable image with everything the results page shows."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    from metrics import fmt_time

    ink = (17, 24, 39)
    mut = (107, 114, 128)
    acc = (29, 78, 216)
    tile = (243, 244, 246)
    line = (229, 231, 235)
    W, M = 1120, 40
    img = Image.new("RGB", (W, 3200), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f_title = _card_font(40, True)
    f_hero = _card_font(64, True)
    f_val = _card_font(30, True)
    f_lab = _card_font(17)
    f_head = _card_font(19, True)
    f_cell = _card_font(19)
    f_note = _card_font(16)

    m = result.get("metrics", {})
    s = m.get("summary", {})
    est = [float(x) for x in result.get("estimated", [])]
    fmt = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731

    y = M
    dr.text((M, y), result.get("swimmer") or "Race results", font=f_title, fill=ink)
    y += 52
    sub = ", ".join(x for x in [(result.get("board") or {}).get("event"), result.get("pool"),
                                Path(result.get("video", "")).name] if x)
    if sub:
        dr.text((M, y), sub, font=f_lab, fill=mut)
        y += 30
    y += 8

    final = result.get("official_time_s") or s.get("final_time_s")
    dr.text((M, y), fmt_time(final) if final else "-", font=f_hero, fill=acc)
    src = "official time" if result.get("finish_source") != "video" else "from the video, can be a few tenths off"
    dr.text((M + 8 + dr.textlength(fmt_time(final) if final else "-", font=f_hero), y + 38),
            f"final time (s), {src}", font=f_lab, fill=mut)
    y += 92

    tiles = []
    for mk in (15, 25, 35):
        v = s.get(f"time_to_{mk}m_s")
        if v is not None:
            tiles.append((f"{v:.2f}", f"Time to {mk} m (s)" + (", estimated" if float(mk) in est else "")))
    tiles += [(fmt(s.get("entry_distance_m"), 1), "Dive entry (m from wall)"),
              (fmt(s.get("stroke_rate_per_min"), 1), "Stroke rate (per min)"),
              (str(s.get("stroke_count", "-")), "Strokes"),
              (fmt(s.get("distance_per_stroke_m")), "Distance per stroke (m)"),
              (fmt(s.get("stroke_index")), "Stroke index"),
              (fmt(s.get("average_speed_mps")), "Average speed (m/s)")]
    cols = 3
    tw = (W - 2 * M - (cols - 1) * 14) // cols
    for k, (v, lab) in enumerate(tiles):
        cx = M + (k % cols) * (tw + 14)
        cy = y + (k // cols) * 96
        dr.rounded_rectangle([cx, cy, cx + tw, cy + 84], 10, fill=tile)
        dr.text((cx + 14, cy + 12), v, font=f_val, fill=ink)
        dr.text((cx + 14, cy + 52), lab, font=f_lab, fill=mut)
    y += ((len(tiles) + cols - 1) // cols) * 96 + 12

    def table(y, title, header, rows, widths):
        dr.text((M, y), title, font=f_head, fill=ink)
        y += 34
        x = M
        for h, w in zip(header, widths):
            dr.text((x, y), h, font=f_head, fill=mut)
            x += w
        y += 30
        dr.line([M, y - 6, W - M, y - 6], fill=line, width=2)
        for row in rows:
            x = M
            for c, w in zip(row, widths):
                dr.text((x, y), str(c), font=f_cell, fill=ink)
                x += w
            y += 30
        return y + 14

    phases = result.get("phases", [])
    if phases:
        rows = [(p["phase"], fmt(p["time_s"]), fmt(p["distance_m"], 1), fmt(p["speed_mps"])) for p in phases]
        y = table(y, "The lap in three phases", ("Phase", "Time (s)", "Distance (m)", "Speed (m/s)"),
                  rows, (500, 180, 190, 170))

    per = result.get("per_second", [])
    curve = m.get("speed_curve", [])
    if curve or per:
        dr.text((M, y), "Speed through the lap", font=f_head, fill=ink)
        y += 34
        ch, cw = 200, W - 2 * M - 50
        x0, y0 = M + 50, y
        pts = [(p["t_s"], p["speed_mps"]) for p in (curve or per)]
        T = max(p[0] for p in pts)
        vmax = max(3.5, max(p[1] for p in pts) + 0.2)
        for g in range(0, int(vmax) + 1):
            gy = y0 + ch - ch * g / vmax
            dr.line([x0, gy, x0 + cw, gy], fill=line)
            dr.text((M, gy - 9), f"{g}", font=f_note, fill=mut)
        xy = [(x0 + cw * t / T, y0 + ch - ch * min(v, vmax) / vmax) for t, v in pts]
        if len(xy) > 1:
            dr.line(xy, fill=acc, width=3)
        for p in per:
            px = x0 + cw * p["t_s"] / T
            py = y0 + ch - ch * min(p["speed_mps"], vmax) / vmax
            dr.ellipse([px - 3, py - 3, px + 3, py + 3], fill=acc)
        for g in range(0, int(T) + 1, 5):
            dr.text((x0 + cw * g / T - 6, y0 + ch + 6), f"{g}", font=f_note, fill=mut)
        dr.text((x0 + cw / 2 - 28, y0 + ch + 24), "seconds", font=f_note, fill=mut)
        y += ch + 40

    splits = m.get("splits", [])
    if splits:
        rows = [(f"{r['distance_m']:g} m", fmt_time(r["time_s"]), r["segment"],
                 fmt(r["segment_time_s"]), fmt(r["segment_speed_mps"])) for r in splits]
        y = table(y, "Splits", ("At", "Time (s)", "Segment", "Seg time (s)", "Speed (m/s)"),
                  rows, (140, 200, 260, 230, 210))

    segs = m.get("segments", [])
    if segs:
        rows = [(r["segment"], fmt(r["speed_mps"]), fmt(r["strokes"], 1), fmt(r["stroke_rate_per_min"], 1),
                 fmt(r["distance_per_stroke_m"]), fmt(r["stroke_index"])) for r in segs]
        y = table(y, "Stroking, segment by segment",
                  ("Segment", "Speed", "Strokes", "Rate/min", "Per stroke (m)", "Stroke index"),
                  rows, (230, 130, 140, 160, 210, 170))

    pics = result.get("pictures", {})
    order = [k for k in ("start", "entry", "breakout", "m15", "m25", "m35", "m45", "finish") if k in pics]
    if order:
        dr.text((M, y), "Key moments", font=f_head, fill=ink)
        y += 34
        pw = (W - 2 * M - 3 * 12) // 4
        row_h = 0
        for k, key in enumerate(order):
            p = Path(workdir) / pics[key]
            if not p.exists():
                continue
            th = Image.open(p)
            th = th.resize((pw, int(pw * th.height / th.width)))
            cx = M + (k % 4) * (pw + 12)
            cy = y + (k // 4) * (th.height + 16)
            img.paste(th, (cx, cy))
            row_h = th.height + 16
        y += ((len(order) + 3) // 4) * row_h + 8

    notes = [v for v in result.get("notes", [])]
    fin = result.get("confidence", {}).get("finish")
    if fin:
        notes.insert(0, "Finish: " + fin)
    for n in notes[:4]:
        dr.text((M, y), "- " + n[:110], font=f_note, fill=mut)
        y += 24
    y += M - 24

    out = Path(workdir) / out_name
    img.crop((0, 0, W, y)).save(out)
    return out_name


def render_speed_video(workdir: str, result: dict, out_name: str = "speed.mp4", progress=None) -> Optional[str]:
    """Write a copy of the working video with the clock, speed and phase in
    the top-left corner from just before the horn to just after the touch."""
    work = Path(workdir)
    proxy = work / "proxy.mp4"
    if not proxy.exists():
        return None
    curve = build_curve(result)
    if curve is None:
        return None
    exe = ffmpeg_binary()
    if exe is None:
        return None
    video = VideoSource(str(proxy))
    fps = result["fps"]
    start = result["events"]["start"]
    T = result["metrics"]["summary"].get("final_time_s")
    if T is None:
        return None
    f0 = max(start - int(1.0 * fps), 0)
    f1 = min(start + int((T + 1.5) * fps), video.frame_count - 1)
    W, H = video.width, video.height

    tmp = work / (out_name + ".part.mp4")
    cmd = [exe, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-",
           "-ss", f"{f0 / fps:.3f}", "-i", str(proxy),
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    scale = W / 1280.0
    box_w, box_h = int(430 * scale), int(120 * scale)
    try:
        for n, i in enumerate(range(f0, f1 + 1)):
            rgb = video.frame(i)
            if rgb is None:
                break
            img = rgb.copy()
            t = (i - start) / fps
            if t >= -0.3:
                tt = max(t, 0.0)
                v = float(curve.derivative(min(tt, T))[0])
                d = float(curve(min(tt, T))[0])
                phase = _phase_at(result, tt) if t <= T else "FINISH"
                if t > T:
                    v = 0.0
                over = img[0:box_h, 0:box_w]
                img[0:box_h, 0:box_w] = (over * 0.25).astype(np.uint8)
                cv2.putText(img, f"{tt:5.1f} s   {phase}", (int(14 * scale), int(34 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)
                speed_txt = f"{v:4.1f} m/s" if t <= T else "  touch"
                cv2.putText(img, speed_txt, (int(14 * scale), int(84 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5 * scale, (80, 220, 255), max(int(3 * scale), 2), cv2.LINE_AA)
                if t <= T:
                    cv2.putText(img, f"{d:4.0f} m", (int(300 * scale), int(84 * scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)
                # a small speed bar under the numbers
                frac = min(max(v / 3.0, 0.0), 1.0)
                y0 = int(100 * scale)
                cv2.rectangle(img, (int(14 * scale), y0), (int(14 * scale) + int(400 * scale), y0 + int(8 * scale)), (70, 70, 70), -1)
                cv2.rectangle(img, (int(14 * scale), y0), (int(14 * scale) + int(400 * scale * frac), y0 + int(8 * scale)), (80, 220, 255), -1)
            proc.stdin.write(img.tobytes())
            if progress and n % 30 == 0:
                progress(n / max(f1 - f0, 1))
        proc.stdin.close()
        proc.wait(timeout=300)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        tmp.unlink(missing_ok=True)
        return None
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return None
    out = work / out_name
    tmp.replace(out)
    if progress:
        progress(1.0)
    return out_name
