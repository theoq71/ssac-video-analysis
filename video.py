"""Video access helpers: frame extraction, audio-based start detection,
and drawing overlays on frames."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoSource:
    """Random and sequential frame access with OpenCV.

    Sequential reads (frame i+1 after frame i) avoid a seek, which matters for
    HEVC phone footage where every seek decodes from the previous keyframe.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_index = 0
        self._last = None

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def frame(self, index: int) -> Optional[np.ndarray]:
        """Return frame `index` as an RGB array, or None past the end."""
        index = int(max(0, min(index, max(self.frame_count - 1, 0))))
        if self._last is not None and self._last[0] == index:
            return self._last[1]
        if index != self._next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_index = index
        ok, bgr = self.cap.read()
        if not ok:
            # some containers report one more frame than they hold
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self.cap.read()
            if not ok:
                return self._last[1] if self._last else None
        self._next_index = index + 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._last = (index, rgb)
        return rgb

    def release(self):
        self.cap.release()


def ffmpeg_binary() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def audio_envelope(path: str, sample_rate: int = 8000, window_s: float = 0.01) -> Optional[tuple[np.ndarray, float]]:
    """Mono RMS envelope of the audio track. Returns (envelope, window_s)."""
    exe = ffmpeg_binary()
    if exe is None:
        return None
    cmd = [exe, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    except Exception:  # noqa: BLE001
        return None
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return None
    win = max(int(sample_rate * window_s), 1)
    n = samples.size // win
    if n == 0:
        return None
    frames = samples[: n * win].reshape(n, win)
    env = np.sqrt((frames ** 2).mean(axis=1))
    return env, window_s


def audio_samples(path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    exe = ffmpeg_binary()
    if exe is None:
        return None
    cmd = [exe, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    except Exception:  # noqa: BLE001
        return None
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples if samples.size else None


def detect_start_signal(path: str, fps: float, search_seconds: float = 30.0,
                        threshold: float = 6.0, sustain_s: float = 0.1) -> Optional[dict]:
    """Guess the frame of the starting horn.

    Starting systems produce a loud tone with energy between roughly 0.7 and
    3.6 kHz, well above the crowd rumble. This looks for the first moment the
    energy in that band jumps to `threshold` times the level of the preceding
    half second and stays there for `sustain_s`.

    Returns {"frame", "time_s", "confidence"} or None.
    """
    sr = 16000
    x = audio_samples(path, sr)
    if x is None:
        return None
    x = x[: int(search_seconds * sr)]
    n, hop = 512, 160
    if len(x) < n + hop * 20:
        return None
    nfr = (len(x) - n) // hop
    w = np.hanning(n)
    frames = np.lib.stride_tricks.as_strided(x, shape=(nfr, n), strides=(x.strides[0] * hop, x.strides[0]))
    spec = np.abs(np.fft.rfft(frames * w, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    band = (freqs >= 700) & (freqs <= 3600)
    e = spec[:, band].sum(axis=1)
    look = 50  # half a second of hops
    base = np.empty(nfr)
    for i in range(nfr):
        prev = e[max(0, i - look):i]
        base[i] = np.median(prev) if prev.size >= 5 else np.median(e[:5])
    ratio = e / (base + 1e-12)
    need = max(int(sustain_s * sr / hop), 1)
    above = ratio > threshold
    for i in range(nfr - need):
        if above[i:i + need].all():
            t = i * hop / sr
            strength = float(np.median(ratio[i:i + need]))
            conf = float(min(strength / 30.0, 1.0))
            return {"frame": int(round(t * fps)), "time_s": t, "confidence": conf}
    return None


def draw_overlay(rgb: np.ndarray, points: list[dict], line: Optional[tuple] = None) -> np.ndarray:
    """Draw calibration points (and the fitted lane line) on a copy of a frame.

    points: dicts with x, y, label, kind ("ref" or "target").
    line: ((x0, y0), (x1, y1)) in pixels.
    """
    img = rgb.copy()
    scale = max(img.shape[1] / 1280.0, 0.6)
    if line is not None:
        (x0, y0), (x1, y1) = line
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), max(int(2 * scale), 1))
    for p in points:
        x, y = int(round(p["x"])), int(round(p["y"]))
        color = (255, 80, 80) if p.get("kind") == "target" else (40, 220, 120)
        cv2.circle(img, (x, y), int(9 * scale), color, -1)
        cv2.circle(img, (x, y), int(9 * scale), (0, 0, 0), max(int(1.5 * scale), 1))
        label = p.get("label", "")
        if label:
            cv2.putText(img, label, (x + int(12 * scale), y - int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * scale, (0, 0, 0), int(4 * scale), cv2.LINE_AA)
            cv2.putText(img, label, (x + int(12 * scale), y - int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * scale, (255, 255, 255), max(int(1.5 * scale), 1), cv2.LINE_AA)
    return img


def draw_timestamp(rgb: np.ndarray, text: str) -> np.ndarray:
    img = rgb.copy()
    scale = max(img.shape[1] / 1280.0, 0.6)
    cv2.rectangle(img, (0, 0), (int(330 * scale), int(44 * scale)), (0, 0, 0), -1)
    cv2.putText(img, text, (int(10 * scale), int(31 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)
    return img


def video_needs_tonemap(path: str) -> bool:
    """True for HDR footage (iPhone HLG or PQ), which decodes washed out."""
    exe = ffmpeg_binary()
    if exe is None:
        return False
    try:
        info = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60).stderr
    except Exception:  # noqa: BLE001
        return False
    return ("arib-std-b67" in info) or ("smpte2084" in info) or ("bt2020" in info)


def make_proxy(src: str, dst: str, progress=None, width: Optional[int] = None) -> tuple[bool, str]:
    """Create an 8-bit, easy-to-scrub H.264 copy of `src` at `dst`.

    HDR sources are tone mapped to standard colour so frames look like they do
    in the Photos app. Frequent keyframes make stepping backwards fast.
    `progress(fraction)` is called as encoding advances.

    Returns (ok, message). Several filter chains are tried in turn, because
    not every ffmpeg build ships the tone-mapping filters; the message says
    which one was used, or why all of them failed.
    """
    exe = ffmpeg_binary()
    if exe is None:
        return False, ("No ffmpeg found. In Terminal run: pip install imageio-ffmpeg "
                       "(or: brew install ffmpeg), then reload.")
    tonemap = video_needs_tonemap(src)
    duration = None
    try:
        info = subprocess.run([exe, "-hide_banner", "-i", str(src)], capture_output=True, text=True, timeout=60).stderr
        for line in info.splitlines():
            if "Duration:" in line:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m, sec = hms.split(":")
                duration = int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:  # noqa: BLE001
        pass

    scale = f",scale={int(width)}:-2" if width else ""
    chains = []
    if tonemap:
        chains.append(("tone mapped",
                       "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
                       "zscale=t=bt709:m=bt709:r=tv,format=yuv420p" + scale))
        chains.append(("approximate colour (no tone-mapping filter in this ffmpeg)",
                       "format=yuv420p,eq=saturation=1.3:contrast=1.05" + scale))
    chains.append(("plain copy", "format=yuv420p" + scale))

    tmp = str(dst) + ".part.mp4"
    errors = []
    for label, vf in chains:
        for vcodec in (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"],
                       ["-c:v", "mpeg4", "-q:v", "4"]):
            cmd = [exe, "-v", "error", "-y", "-i", str(src), "-vf", vf, *vcodec, "-g", "12",
                   "-vsync", "passthrough", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                   "-progress", "pipe:1", "-nostats", tmp]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stdout:
                    if progress and duration and line.startswith("out_time_ms="):
                        try:
                            done = int(line.split("=")[1]) / 1e6
                            progress(min(done / duration, 1.0))
                        except ValueError:
                            pass
                err = proc.stderr.read()
                proc.wait()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}/{vcodec[1]}: {e}")
                continue
            if proc.returncode == 0 and Path(tmp).exists() and Path(tmp).stat().st_size > 0:
                Path(tmp).replace(dst)
                if progress:
                    progress(1.0)
                return True, f"Working copy made ({label}, {vcodec[1]})."
            Path(tmp).unlink(missing_ok=True)
            errors.append(f"{label}/{vcodec[1]}: " + (err.strip().splitlines() or ["exit " + str(proc.returncode)])[-1])
    return False, "ffmpeg at " + exe + " could not convert the video. " + " | ".join(errors[-3:])


def list_videos(folder: Path) -> list[Path]:
    exts = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file())
