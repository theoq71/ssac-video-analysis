# SSAC Race Analysis

A web page for coaches. Upload a race clip from your phone, tap your swimmer
once on the start frame, and a few minutes later read the results: final
time, time to 15 m, 25 m and 35 m, dive entry distance, stroke rate, stroke
count, distance per stroke, stroke index, and speed through the lap.

Nothing else is marked by hand. The app listens for the horn, follows the
lane ropes and the swimmer's white water through the camera's panning, reads
the 5 m and 15 m colour changes on the ropes, finds the breakout and the
finish, and measures the stroke rhythm from the spray.

## Running it

One computer runs the app; everyone else uses a phone browser.

```
cd "SSAC Video Analysis"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

It prints the address to open, like `http://192.168.1.20:8000`. Phones on the
same wifi open that address. To reach it from anywhere, put it behind a
tunnel or a small cloud host later; the app itself does not change.

Optional: `brew install tesseract` lets the app read the official time off
the results board when the clip ends on it.

## What a coach does

1. Open the page, choose the video, type the swimmer's name if you want it
   on the results, press Upload.
2. Wait for the start frame (about as long as the clip). Tap the swimmer.
   Press Analyse.
3. Two to four minutes later the results page appears. The link keeps
   working, so it can be shared or revisited.

The results page shows the lap in three phases (dive, underwater, swim),
the speed at every second, a copy of the video with the clock, current
speed and phase drawn in the top corner (downloadable, for sharing), and
the frame it chose for each key moment so a glance tells you whether to
trust a number. If you have the official time, type it in and everything
is recalculated, including the speed video.

## How good is it

Tested on one 50 m butterfly filmed handheld from the side. Against the
scoreboard's 26.72: the app's finish was 26.60 from the video alone, the
15 m time (8.57 s) matched a frame-by-frame check, the stroke rate (54 to 55
per minute) matched a manual count, and the dive entry (1.3 m) is within
about half a metre. Splits at distances with no rope mark (25 m and 35 m in
this pool) are interpolated and say so.

What breaks it: a camera that swings away before the touch, lanes where the
rope colour changes are not at 5 m and 15 m from the wall (set the pool
profile in `autorace.py`), and a swimmer whose lane is hidden behind other
swimmers' splash for long stretches.

## Filming

- Stand as high as you can and as square to the lane as possible.
- Start recording before the horn; keep the swimmer roughly centred; hold on
  the wall for a second after the touch.
- If the results board is nearby, film it for a few seconds at the end.
- 1080p at 30 or 60 frames per second is plenty.

## Files

- `server.py` - the web app
- `autorace.py` - the pipeline from video to numbers
- `engine.py` - per-frame tracking of ropes, lane and white water
- `ropes.py` - rope detection and the colour-change marks
- `resolve.py` - landmarks, crossings, entry, finish
- `scoreboard.py` - optional reading of the results board
- `annotate.py` - phases, per-second speeds, the annotated speed video
- `metrics.py` - splits, stroke rate, distance per stroke, stroke index, speed curve
- `video.py` - frame access, horn detection, working copies of HDR phone video
- `jobs/` - one folder per uploaded race (video, working copy, results)
