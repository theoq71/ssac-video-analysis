"""SSAC race analysis: the web app.

    python server.py            (then open http://<this computer>:8000 on any phone on the network)

A coach uploads a race clip, taps their swimmer on the start frame, and
gets a results page a few minutes later. Everything else is automatic.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import (Flask, Response, abort, jsonify, redirect, render_template_string, request,
                   send_from_directory, url_for)

from annotate import per_second_speeds, phase_summary, render_results_card, render_speed_video
from autorace import POOL_PROFILES, analyze_race, review_frames, start_frame_for
from metrics import RaceInput, analyze, fmt_time
from video import VideoSource, make_proxy

APP_DIR = Path(__file__).resolve().parent
JOBS = APP_DIR / "jobs"
JOBS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3

_lock = threading.Lock()
_status: dict[str, dict] = {}


def job_dir(job_id: str) -> Path:
    if not job_id.replace("-", "").isalnum():
        abort(404)
    d = JOBS / job_id
    if not d.exists():
        abort(404)
    return d


def set_status(job_id: str, **kw):
    with _lock:
        _status.setdefault(job_id, {}).update(kw)
        (JOBS / job_id / "status.json").write_text(json.dumps(_status[job_id]))


def get_status(job_id: str) -> dict:
    with _lock:
        st = _status.get(job_id)
    if st is None:
        p = JOBS / job_id / "status.json"
        st = json.loads(p.read_text()) if p.exists() else {"stage": "unknown", "frac": 0}
    return st


# ---------------------------------------------------------------------------
# background work
# ---------------------------------------------------------------------------

def prepare_job(job_id: str):
    """Make the working copy and the start frame the coach will tap on."""
    d = JOBS / job_id
    meta = json.loads((d / "meta.json").read_text())
    try:
        set_status(job_id, stage="Preparing the video", frac=0.0, phase="prepare")
        ok, msg = make_proxy(str(d / meta["video"]), str(d / "proxy.mp4"),
                             progress=lambda f: set_status(job_id, stage="Preparing the video", frac=float(f)), width=1280)
        if not ok:
            set_status(job_id, stage="failed", error=msg, phase="error")
            return
        src = VideoSource(str(d / meta["video"]))
        fps = src.fps
        src.release()
        set_status(job_id, stage="Listening for the horn", frac=0.0)
        st = start_frame_for(str(d / meta["video"]), fps)
        video = VideoSource(str(d / "proxy.mp4"))
        frame = video.frame(st["frame"])
        cv2.imwrite(str(d / "start.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        meta["start"] = st
        meta["fps"] = fps
        meta["frame_w"], meta["frame_h"] = int(frame.shape[1]), int(frame.shape[0])
        (d / "meta.json").write_text(json.dumps(meta))
        set_status(job_id, stage="ready for the tap", frac=1.0, phase="tap")
    except Exception as e:  # noqa: BLE001
        set_status(job_id, stage="failed", error=str(e), phase="error")


def run_analysis(job_id: str, tap: tuple[float, float]):
    d = JOBS / job_id
    meta = json.loads((d / "meta.json").read_text())
    try:
        def prog(stage, frac):
            set_status(job_id, stage=stage, frac=float(frac), phase="analysis")
        result = analyze_race(str(d / meta["video"]), tap=tap, workdir=str(d), swimmer_name=meta.get("swimmer", ""),
                              pool=meta.get("pool", "50 m (long course)"), progress=prog)
        result["pictures"] = review_frames(str(d / meta["video"]), str(d), result)
        set_status(job_id, stage="Making the speed video", frac=0.0, phase="analysis")
        result["speed_video"] = render_speed_video(str(d), result,
                                                   progress=lambda f: set_status(job_id, stage="Making the speed video", frac=float(f)))
        result["swimmer"] = meta.get("swimmer", "")
        result["pool"] = meta.get("pool", "")
        result["tap"] = list(tap)
        result["results_image"] = render_results_card(str(d), result)
        (d / "result.json").write_text(json.dumps(result, indent=2, default=float))
        set_status(job_id, stage="done", frac=1.0, phase="done")
    except Exception as e:  # noqa: BLE001
        import traceback
        (d / "error.txt").write_text(traceback.format_exc())
        set_status(job_id, stage="failed", error=str(e), phase="error")


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

BASE_CSS = """
:root { --ink:#17202a; --muted:#5b6672; --line:#e3e8ee; --accent:#1f6fd0; --bg:#f6f8fb; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color:var(--ink); background:var(--bg); }
main { max-width: 760px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 28px 0 8px; }
p, li { line-height: 1.5; }
.card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px; margin:12px 0; }
.muted { color: var(--muted); }
label { display:block; font-weight:600; margin: 12px 0 4px; }
input[type=text], select { width:100%; font-size:1rem; padding:10px; border:1px solid var(--line); border-radius:8px; }
input[type=file] { font-size:1rem; }
button, .btn { display:inline-block; font-size:1.05rem; padding:12px 18px; border:0; border-radius:10px; background:var(--accent); color:#fff; text-decoration:none; cursor:pointer; }
button.secondary { background:#e9eef5; color:var(--ink); }
.big { font-size:2rem; font-weight:700; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }
.stat { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; }
.stat .v { font-size:1.6rem; font-weight:700; }
.stat .l { color:var(--muted); font-size:0.85rem; }
table { width:100%; border-collapse: collapse; font-size:0.95rem; }
th, td { text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; font-size:0.85rem; }
.pics { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }
.pics img { width:100%; border-radius:8px; border:1px solid var(--line); }
.pics figcaption { font-size:0.85rem; color:var(--muted); margin-top:4px; }
.bar { height:10px; background:#e3e8ee; border-radius:6px; overflow:hidden; }
.bar > div { height:100%; background:var(--accent); width:0; transition: width .4s; }
.tapwrap { position:relative; }
.tapwrap img { width:100%; display:block; border-radius:10px; }
.dot { position:absolute; width:22px; height:22px; margin:-11px 0 0 -11px; border-radius:50%; border:3px solid #fff; background:var(--accent); box-shadow:0 0 0 2px var(--accent); display:none; }
.note { font-size:0.9rem; color:var(--muted); }
.warn { background:#fff7e6; border-color:#f3d9a4; }
"""

INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SSAC Race Analysis</title><style>{{ css|safe }}</style></head><body><main>
<h1>SSAC Race Analysis</h1>
<p class="muted">Upload a race video from your phone. You will tap your swimmer once, and the rest is automatic.</p>
<form class="card" method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
  <label>Race video</label>
  <input type="file" name="video" accept="video/*" required>
  <label>Swimmer (optional)</label>
  <input type="text" name="swimmer" placeholder="Name, so the results are labelled">
  <label>Pool</label>
  <select name="pool">{% for p in pools %}<option>{{p}}</option>{% endfor %}</select>
  <p class="note">Film from the side, as high as you can, and start recording before the horn. Filming the results board afterwards lets the app read the official time.</p>
  <button type="submit">Upload and analyse</button>
</form>
{% if recent %}
<h2>Recent races</h2>
<div class="card">{% for r in recent %}<p><a href="{{ url_for('results', job_id=r.id) }}">{{ r.label }}</a> <span class="muted">{{ r.when }}</span></p>{% endfor %}</div>
{% endif %}
</main></body></html>
"""

TAP_HTML = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tap your swimmer</title><style>{{ css|safe }}</style></head><body><main>
<h1>Tap your swimmer</h1>
<div id="prep" class="card">
  <p id="stage">Preparing the video</p>
  <div class="bar"><div id="bar"></div></div>
  <p class="note">This takes about as long as the clip itself.</p>
</div>
<div id="tap" class="card" style="display:none">
  <p>This is the moment of the start. <b>Tap your swimmer</b> (on the block is fine), then press Analyse.</p>
  <div class="tapwrap" id="wrap"><img id="frame" alt="start frame"><div class="dot" id="dot"></div></div>
  <p style="margin-top:12px"><button id="go" disabled>Analyse this swimmer</button> <button class="secondary" id="clear">Clear</button></p>
</div>
<div id="run" class="card" style="display:none">
  <p id="stage2">Working</p>
  <div class="bar"><div id="bar2"></div></div>
  <p class="note">Two to four minutes. You can leave this page and come back; the link stays valid.</p>
</div>
<div id="err" class="card warn" style="display:none"></div>
<script>
const jobId = "{{ job_id }}";
let tapPt = null;
const stageEl = document.getElementById('stage'), bar = document.getElementById('bar');
const stage2 = document.getElementById('stage2'), bar2 = document.getElementById('bar2');
function show(id){ for (const k of ['prep','tap','run']) document.getElementById(k).style.display = (k===id?'block':'none'); }
async function poll(){
  const r = await fetch('/jobs/'+jobId+'/status'); const s = await r.json();
  if (s.phase === 'error'){ document.getElementById('err').style.display='block'; document.getElementById('err').textContent = 'Something went wrong: ' + s.error; return; }
  if (s.phase === 'prepare' || !s.phase){ stageEl.textContent = s.stage; bar.style.width = Math.round((s.frac||0)*100)+'%'; setTimeout(poll, 1500); return; }
  if (s.phase === 'tap'){ show('tap'); document.getElementById('frame').src = '/jobs/'+jobId+'/start.jpg?'+Date.now(); return; }
  if (s.phase === 'analysis'){ show('run'); stage2.textContent = s.stage; bar2.style.width = Math.round((s.frac||0)*100)+'%'; setTimeout(poll, 2000); return; }
  if (s.phase === 'done'){ window.location = '/jobs/'+jobId+'/results'; return; }
  setTimeout(poll, 1500);
}
document.getElementById('wrap').addEventListener('click', ev => {
  const img = document.getElementById('frame'); const rect = img.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width, y = (ev.clientY - rect.top) / rect.height;
  tapPt = {x: x * img.naturalWidth, y: y * img.naturalHeight};
  const dot = document.getElementById('dot'); dot.style.left = (x*100)+'%'; dot.style.top = (y*100)+'%'; dot.style.display = 'block';
  document.getElementById('go').disabled = false;
});
document.getElementById('clear').addEventListener('click', () => { tapPt = null; document.getElementById('dot').style.display='none'; document.getElementById('go').disabled = true; });
document.getElementById('go').addEventListener('click', async () => {
  if (!tapPt) return;
  document.getElementById('go').disabled = true;
  await fetch('/jobs/'+jobId+'/tap', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(tapPt)});
  show('run'); poll();
});
poll();
</script>
</main></body></html>
"""

RESULTS_HTML = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results</title><style>{{ css|safe }}</style></head><body><main>
<h1>{{ title }}</h1>
<p class="muted">{{ subtitle }}</p>
<div class="grid">
  <div class="stat"><div class="v">{{ s.final }}</div><div class="l">Final time (s){% if finish_official %}, official{% else %}, from the video{% endif %}</div></div>
  {% for m in splits %}<div class="stat"><div class="v">{{ m.t }}</div><div class="l">Time to {{ m.d }} m (s){% if m.est %}, estimated{% endif %}</div></div>{% endfor %}
  <div class="stat"><div class="v">{{ s.entry }}</div><div class="l">Dive entry (m from wall)</div></div>
  <div class="stat"><div class="v">{{ s.rate }}</div><div class="l">Stroke rate (per min)</div></div>
  <div class="stat"><div class="v">{{ s.count }}</div><div class="l">Strokes</div></div>
  <div class="stat"><div class="v">{{ s.dps }}</div><div class="l">Distance per stroke (m)</div></div>
  <div class="stat"><div class="v">{{ s.si }}</div><div class="l">Stroke index</div></div>
  <div class="stat"><div class="v">{{ s.speed }}</div><div class="l">Average speed (m/s)</div></div>
</div>

<h2>The lap in three phases</h2>
<div class="card"><table><tr><th>Phase</th><th>From (s)</th><th>To (s)</th><th>Time (s)</th><th>Distance (m)</th><th>Speed (m/s)</th></tr>
{% for p in phase_rows %}<tr><td>{{ p.name }}{% if p.note %} <span class="note">({{ p.note }})</span>{% endif %}</td><td>{{ p.t0 }}</td><td>{{ p.t1 }}</td><td>{{ p.dt }}</td><td>{{ p.d }}</td><td>{{ p.v }}</td></tr>{% endfor %}
</table></div>

{% if speed_video %}
<h2>The race with live speed</h2>
<div class="card">
  <video controls playsinline preload="metadata" style="width:100%; border-radius:8px" src="{{ url_for('job_file', job_id=job_id, fname=speed_video) }}"></video>
  <p class="note" style="margin-bottom:0">The clock, the current speed and the phase are drawn in the top corner.
  <a href="{{ url_for('job_file', job_id=job_id, fname=speed_video) }}" download>Download this video</a> to share it.</p>
</div>
{% endif %}

{% if not finish_official %}
<form class="card" method="post" action="{{ url_for('set_official', job_id=job_id) }}">
  <p class="note">The finish is read from the video and can be a few tenths off. If you have the official time, enter it and everything is recalculated.</p>
  <input type="text" name="official" inputmode="decimal" placeholder="e.g. 26.72" style="max-width:200px"> <button type="submit">Use official time</button>
</form>
{% endif %}

<h2>Speed through the lap</h2>
<div class="card"><canvas id="chart" width="700" height="320" style="width:100%; height:auto"></canvas></div>

{% if per_second %}
<h2>Speed at each second</h2>
<div class="card" style="overflow-x:auto"><table><tr><th>Second</th>{% for r in per_second %}<td>{{ r.t_s }}</td>{% endfor %}</tr>
<tr><th>Speed (m/s)</th>{% for r in per_second %}<td>{{ '%.1f'|format(r.speed_mps) }}</td>{% endfor %}</tr>
<tr><th>Distance (m)</th>{% for r in per_second %}<td>{{ '%.0f'|format(r.distance_m) }}</td>{% endfor %}</tr>
</table><p class="note">Approximate: read off the smooth distance curve fitted through the measured splits.</p></div>
{% endif %}

<h2>Splits</h2>
<div class="card"><table><tr><th>At</th><th>Time (s)</th><th>Segment</th><th>Segment time (s)</th><th>Speed (m/s)</th></tr>
{% for r in split_rows %}<tr><td>{{ r.at }} m</td><td>{{ r.t }}</td><td>{{ r.seg }}</td><td>{{ r.st }}</td><td>{{ r.v }}</td></tr>{% endfor %}
</table></div>

{% if seg_rows %}
<h2>Stroking, segment by segment</h2>
<div class="card"><table><tr><th>Segment</th><th>Speed (m/s)</th><th>Strokes</th><th>Rate (per min)</th><th>Per stroke (m)</th><th>Stroke index</th></tr>
{% for r in seg_rows %}<tr><td>{{ r.seg }}</td><td>{{ r.v }}</td><td>{{ r.n }}</td><td>{{ r.rate }}</td><td>{{ r.dps }}</td><td>{{ r.si }}</td></tr>{% endfor %}
</table></div>
{% endif %}

<h2>Check the key moments</h2>
<p class="note">Each picture is the frame the app chose. If one looks wrong, the number next to it is off by about that much.</p>
<div class="pics">
{% for p in pics %}<figure style="margin:0"><img src="{{ p.src }}" alt="{{ p.label }}"><figcaption>{{ p.label }}: {{ p.note }}</figcaption></figure>{% endfor %}
</div>

{% if notes %}<h2>Notes</h2><div class="card">{% for n in notes %}<p class="note">{{ n }}</p>{% endfor %}</div>{% endif %}

<p style="margin-top:24px">{% if results_image %}<a class="btn" href="{{ url_for('job_file', job_id=job_id, fname=results_image) }}" download>Download results image</a> {% endif %}<a class="btn{% if results_image %} secondary{% endif %}" href="{{ url_for('csv_summary', job_id=job_id) }}">Download summary (CSV)</a> <a class="btn secondary" href="{{ url_for('index') }}">Analyse another race</a></p>
<script>
const curve = {{ curve|tojson }}; const marks = {{ mark_times|tojson }};
const c = document.getElementById('chart'), ctx = c.getContext('2d');
if (curve.length > 1) {
  const W = c.width, H = c.height, pad = {l:44, r:12, t:12, b:28};
  const tmax = curve[curve.length-1].t, vmax = Math.max(...curve.map(p=>p.v)) * 1.1;
  const X = t => pad.l + (W-pad.l-pad.r) * t / tmax, Y = v => H - pad.b - (H-pad.t-pad.b) * v / vmax;
  ctx.strokeStyle = '#e3e8ee'; ctx.lineWidth = 1;
  for (let v = 0; v <= vmax; v += 0.5) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W-pad.r, Y(v)); ctx.stroke(); ctx.fillStyle='#5b6672'; ctx.font='12px sans-serif'; ctx.fillText(v.toFixed(1)+' m/s', 2, Y(v)+4); }
  for (const m of marks) { ctx.strokeStyle='#c8d0da'; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(X(m.t), pad.t); ctx.lineTo(X(m.t), H-pad.b); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle='#5b6672'; ctx.fillText(m.label, X(m.t)+3, pad.t+12); }
  ctx.strokeStyle = '#1f6fd0'; ctx.lineWidth = 2.5; ctx.beginPath();
  curve.forEach((p,i) => { if (i===0) ctx.moveTo(X(p.t), Y(p.v)); else ctx.lineTo(X(p.t), Y(p.v)); }); ctx.stroke();
  ctx.fillStyle='#5b6672'; for (let t = 0; t <= tmax; t += 5) ctx.fillText(t+' s', X(t)-8, H-8);
}
</script>
</main></body></html>
"""


def _recent():
    out = []
    for d in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        rj = d / "result.json"
        if rj.exists():
            try:
                r = json.loads(rj.read_text())
                lab = r.get("swimmer") or Path(r.get("video", "")).name
                out.append({"id": d.name, "label": lab, "when": time.strftime("%b %d %H:%M", time.localtime(rj.stat().st_mtime))})
            except Exception:  # noqa: BLE001
                pass
    return out


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, css=BASE_CSS, pools=list(POOL_PROFILES), recent=_recent())


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        abort(400)
    job_id = uuid.uuid4().hex[:10]
    d = JOBS / job_id
    d.mkdir()
    ext = Path(f.filename).suffix.lower() or ".mp4"
    name = "race" + ext
    f.save(str(d / name))
    meta = {"video": name, "swimmer": request.form.get("swimmer", "").strip(),
            "pool": request.form.get("pool", "50 m (long course)"), "original_name": f.filename}
    (d / "meta.json").write_text(json.dumps(meta))
    set_status(job_id, stage="Queued", frac=0.0, phase="prepare")
    threading.Thread(target=prepare_job, args=(job_id,), daemon=True).start()
    return redirect(url_for("tap", job_id=job_id))


@app.route("/jobs/<job_id>/tap")
def tap(job_id):
    job_dir(job_id)
    return render_template_string(TAP_HTML, css=BASE_CSS, job_id=job_id)


@app.route("/jobs/<job_id>/tap", methods=["POST"])
def tap_post(job_id):
    d = job_dir(job_id)
    data = request.get_json(force=True)
    x, y = float(data["x"]), float(data["y"])
    threading.Thread(target=run_analysis, args=(job_id, (x, y)), daemon=True).start()
    set_status(job_id, stage="Starting", frac=0.0, phase="analysis")
    return jsonify({"ok": True})


@app.route("/jobs/<job_id>/status")
def status(job_id):
    job_dir(job_id)
    return jsonify(get_status(job_id))


@app.route("/jobs/<job_id>/<path:fname>")
def job_file(job_id, fname):
    d = job_dir(job_id)
    if fname not in ("start.jpg", "speed.mp4", "results.png") and not fname.startswith("review_"):
        abort(404)
    return send_from_directory(str(d), fname, conditional=True)


def _rows(result: dict):
    m = result["metrics"]
    s = m.get("summary", {})
    fmt = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    splits = []
    for mk in (15, 25, 35):
        key = f"time_to_{mk}m_s"
        if s.get(key) is not None:
            splits.append({"d": mk, "t": fmt_time(s[key]), "est": float(mk) in [float(x) for x in result.get("estimated", [])]})
    stats = {
        "final": fmt_time(result.get("official_time_s") or s.get("final_time_s")),
        "entry": fmt(s.get("entry_distance_m")),
        "rate": fmt(s.get("stroke_rate_per_min"), 1),
        "count": s.get("stroke_count", "-"),
        "dps": fmt(s.get("distance_per_stroke_m")),
        "si": fmt(s.get("stroke_index")),
        "speed": fmt(s.get("average_speed_mps")),
    }
    split_rows = [{"at": f"{r['distance_m']:g}", "t": fmt_time(r["time_s"]), "seg": r["segment"],
                   "st": fmt(r["segment_time_s"]), "v": fmt(r["segment_speed_mps"])} for r in m.get("splits", [])]
    seg_rows = [{"seg": r["segment"], "v": fmt(r["speed_mps"]), "n": fmt(r["strokes"], 1),
                 "rate": fmt(r["stroke_rate_per_min"], 1), "dps": fmt(r["distance_per_stroke_m"]), "si": fmt(r["stroke_index"])}
                for r in m.get("segments", [])]
    curve = [{"t": round(p["t_s"], 2), "v": round(p["speed_mps"], 3)} for p in m.get("speed_curve", [])]
    mark_times = [{"t": r["time_s"], "label": f"{r['distance_m']:g} m"} for r in m.get("splits", [])]
    return stats, splits, split_rows, seg_rows, curve, mark_times


@app.route("/jobs/<job_id>/results")
def results(job_id):
    d = job_dir(job_id)
    rj = d / "result.json"
    if not rj.exists():
        return redirect(url_for("tap", job_id=job_id))
    result = json.loads(rj.read_text())
    stats, splits, split_rows, seg_rows, curve, mark_times = _rows(result)
    conf = result.get("confidence", {})
    labels = {"start": "Start", "entry": "Dive entry", "breakout": "Breakout", "m15": "15 m", "m25": "25 m",
              "m35": "35 m", "m45": "45 m", "finish": "Finish"}
    pics = []
    for key, fname in result.get("pictures", {}).items():
        pics.append({"src": url_for("job_file", job_id=job_id, fname=fname), "label": labels.get(key, key),
                     "note": conf.get(key, "")})
    board = result.get("board") or {}
    event = board.get("event") if board else None
    title = result.get("swimmer") or "Race results"
    subtitle = ", ".join(x for x in [event, result.get("pool"), Path(result.get("video", "")).name] if x)
    fmt = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    phase_rows = [{"name": p["phase"], "t0": fmt(p["from_s"]), "t1": fmt(p["to_s"]), "dt": fmt(p["time_s"]),
                   "d": fmt(p["distance_m"], 1), "v": fmt(p["speed_mps"]), "note": p.get("note", "")}
                  for p in result.get("phases", [])]
    per_second = result.get("per_second", [])
    return render_template_string(RESULTS_HTML, css=BASE_CSS, job_id=job_id, title=title, subtitle=subtitle, s=stats,
                                  splits=splits, split_rows=split_rows, seg_rows=seg_rows, curve=curve,
                                  mark_times=mark_times, pics=pics, notes=result.get("notes", []),
                                  phase_rows=phase_rows, per_second=per_second,
                                  speed_video=result.get("speed_video"),
                                  results_image=result.get("results_image"),
                                  finish_official=result.get("finish_source") != "video")


@app.route("/jobs/<job_id>/official", methods=["POST"])
def set_official(job_id):
    d = job_dir(job_id)
    result = json.loads((d / "result.json").read_text())
    try:
        t = request.form.get("official", "").strip().replace(",", ".")
        if ":" in t:
            mm, ss = t.split(":")
            secs = int(mm) * 60 + float(ss)
        else:
            secs = float(t)
    except ValueError:
        return redirect(url_for("results", job_id=job_id))
    fps = result["fps"]
    start = result["events"]["start"]
    result["events"]["finish"] = int(round(start + secs * fps))
    result["official_time_s"] = secs
    result["finish_source"] = "official time"
    result["confidence"]["finish"] = f"official time {secs:.2f} entered by the coach"
    # recompute the metrics with the same strokes
    strokes = [start + int(round(t * fps)) for t in result["metrics"].get("strokes", {}).get("times_s", [])]
    inp = RaceInput(fps=fps, course_length=POOL_PROFILES.get(result.get("pool"), POOL_PROFILES["50 m (long course)"])["length"],
                    events=result["events"], strokes=strokes, entry_distance_m=result.get("entry_distance_m"))
    m = analyze(inp)
    m["rhythm"] = result["metrics"].get("rhythm", [])
    if m["rhythm"]:
        m["summary"]["stroke_rate_per_min"] = result["metrics"]["summary"].get("stroke_rate_per_min")
        m["summary"]["stroke_count"] = len(strokes)
    result["metrics"] = m
    result["pictures"] = review_frames(str(d / json.loads((d / "meta.json").read_text())["video"]), str(d), result)
    result["phases"] = phase_summary(result)
    result["per_second"] = per_second_speeds(result)
    result["speed_video"] = render_speed_video(str(d), result)
    result["results_image"] = render_results_card(str(d), result)
    (d / "result.json").write_text(json.dumps(result, indent=2, default=float))
    return redirect(url_for("results", job_id=job_id))


@app.route("/jobs/<job_id>/summary.csv")
def csv_summary(job_id):
    d = job_dir(job_id)
    result = json.loads((d / "result.json").read_text())
    s = result["metrics"]["summary"]
    row = {"swimmer": result.get("swimmer", ""), "event": (result.get("board") or {}).get("event"),
           "pool": result.get("pool"), "video": Path(result.get("video", "")).name, "finish_source": result.get("finish_source")}
    row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    for sg in result["metrics"].get("segments", []):
        tag = sg["segment"].replace(" m", "m").replace("-", "_to_").replace(".0", "")
        row[f"{tag}_rate"] = None if sg["stroke_rate_per_min"] is None else round(sg["stroke_rate_per_min"], 2)
        row[f"{tag}_dps"] = None if sg["distance_per_stroke_m"] is None else round(sg["distance_per_stroke_m"], 3)
        row[f"{tag}_si"] = None if sg["stroke_index"] is None else round(sg["stroke_index"], 3)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(row))
    w.writeheader()
    w.writerow(row)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={job_id}_summary.csv"})


if __name__ == "__main__":
    import socket
    host = socket.gethostbyname(socket.gethostname())
    print(f"Open http://{host}:8000 on a phone on the same network, or http://localhost:8000 here.")
    app.run(host="0.0.0.0", port=8000, threaded=True)
