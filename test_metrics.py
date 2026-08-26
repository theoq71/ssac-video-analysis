import math

import numpy as np

from metrics import (LaneCalibration, MonotoneCubic, RaceInput, RefPoint,
                     analyze, split_marks_for_course)


def synthetic_projection(meters, f=1200.0, z0=20.0, angle=0.35):
    """Pinhole camera looking along a lane from the side at an angle.

    Points along the lane at distance `meters` from the wall map to image x
    with perspective foreshortening.
    """
    # world: lane along X, camera at distance z0 rotated by `angle`
    X = meters - 25.0
    Z = z0 + X * math.sin(angle)
    Xc = X * math.cos(angle)
    x = 960 + f * Xc / Z
    y = 540 + f * 3.0 / Z
    return x, y


def test_projective_calibration_recovers_distance():
    refs = [RefPoint(*synthetic_projection(m), meters=m) for m in (0.0, 5.0, 15.0)]
    cal = LaneCalibration(refs)
    assert cal.mode == "projective"
    for m in (2.0, 3.7, 8.0, 12.0, 25.0):
        x, y = synthetic_projection(m)
        assert abs(cal.distance_at(x, y) - m) < 0.02, (m, cal.distance_at(x, y))
    assert max(abs(r) for r in cal.residuals()) < 1e-6


def test_linear_calibration_with_two_points():
    cal = LaneCalibration([RefPoint(100, 500, 0.0), RefPoint(700, 520, 15.0)])
    assert cal.mode == "linear"
    assert abs(cal.distance_at(300, 510) - 5.0) < 0.05
    # points off the lane line are projected onto it
    assert abs(cal.distance_at(300, 560) - 5.0) < 0.1


def test_calibration_roundtrip():
    cal = LaneCalibration([RefPoint(1, 2, 0.0, "wall"), RefPoint(50, 4, 5.0), RefPoint(150, 8, 15.0)])
    cal2 = LaneCalibration.from_dict(cal.to_dict())
    assert abs(cal2.distance_at(90, 5) - cal.distance_at(90, 5)) < 1e-9


def test_monotone_cubic_interpolates_knots_and_is_monotone():
    t = [0, 0.8, 6.5, 12.0, 17.5, 26.0]
    d = [0, 3.2, 15, 25, 35, 50]
    c = MonotoneCubic(t, d)
    for ti, di in zip(t, d):
        assert abs(c(ti)[0] - di) < 1e-9
    xs = np.linspace(0, 26, 500)
    ys = c(xs)
    assert np.all(np.diff(ys) >= -1e-9)
    assert np.all(c.derivative(xs) >= -1e-9)
    # derivative matches finite difference
    v = c.derivative(10.0)[0]
    fd = (c(10.001)[0] - c(9.999)[0]) / 0.002
    assert abs(v - fd) < 1e-3


def test_full_analysis_50m():
    fps = 30.0
    start = 60
    # a 26.8 s 50 m swim: entry 0.8 s at 3.2 m, 15 m at 6.4 s, 25 m at 12.0 s,
    # 35 m at 17.6 s, finish 26.8 s; breakout at 4.0 s
    ev = {
        "start": start,
        "entry": start + round(0.8 * fps),
        "breakout": start + round(4.0 * fps),
        "m15": start + round(6.4 * fps),
        "m25": start + round(12.0 * fps),
        "m35": start + round(17.6 * fps),
        "finish": start + round(26.8 * fps),
    }
    # strokes every 1.2 s from 4.4 s to 26.0 s
    strokes = [start + round(t * fps) for t in np.arange(4.4, 26.1, 1.2)]
    res = analyze(RaceInput(fps=fps, course_length=50, events=ev, strokes=strokes, entry_distance_m=3.2))
    assert res["ok"]
    assert res["missing"] == []
    s = res["summary"]
    assert abs(s["final_time_s"] - 26.8) < 1 / fps
    assert abs(s["time_to_15m_s"] - 6.4) < 1 / fps
    assert abs(s["time_to_25m_s"] - 12.0) < 1 / fps
    assert abs(s["time_to_35m_s"] - 17.6) < 1 / fps
    assert s["entry_distance_m"] == 3.2
    assert 3.2 < s["breakout_distance_m"] < 15
    assert len(res["splits"]) == 4
    assert [sp["distance_m"] for sp in res["splits"]] == [15, 25, 35, 50]
    assert len(res["segments"]) == 4
    seg = res["segments"][1]  # 15-25 m
    assert abs(seg["stroke_rate_per_min"] - 50.0) < 1.5  # 60 / 1.2
    assert abs(seg["distance_per_stroke_m"] - 10.0 / (5.6 / 1.2)) < 0.15
    assert abs(seg["stroke_index"] - seg["speed_mps"] * seg["distance_per_stroke_m"]) < 1e-9
    assert s["stroke_count"] == len(strokes)
    assert abs(s["stroke_rate_per_min"] - 50.0) < 1.5
    assert len(res["speed_curve"]) > 200
    rates = [r["rate_per_min"] for r in res["stroke_rate_series"]]
    assert all(abs(r - 50) < 2 for r in rates)
    assert all(r["speed_mps"] is not None for r in res["stroke_rate_series"] if 4.4 < r["t_s"] < 26)


def test_partial_marks_still_produce_what_they_can():
    ev = {"start": 0, "m15": 190, "finish": 800}
    res = analyze(RaceInput(fps=30, course_length=50, events=ev, strokes=[]))
    assert res["ok"]
    assert set(res["missing"]) == {"m25", "m35"}
    assert res["summary"]["time_to_15m_s"] == 190 / 30
    assert res["segments"] == []
    assert len(res["speed_curve"]) > 0


def test_short_course_marks():
    assert split_marks_for_course(50) == [15, 25, 35]
    assert split_marks_for_course(25) == [15]
    assert split_marks_for_course(22.86) == [15]


def test_missing_start():
    res = analyze(RaceInput(fps=30, events={"finish": 100}))
    assert not res["ok"] and res["missing"] == ["start"]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("ok  ", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, repr(e))
    sys.exit(1 if failed else 0)
