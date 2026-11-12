"""Unit tests for fwc_controller/fwc_controller/pure_pursuit.py."""

import numpy as np
import pytest

import sim2d
from sim2d import pp


@pytest.fixture(scope='module', params=['track', 'levine'])
def path(request):
    _, ctl = sim2d.experiment_params()
    return sim2d.load_path(request.param, ctl)


def test_path_is_one_closed_lap_at_even_spacing(path):
    p, speeds = path
    steps = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    assert steps.max() < 0.08 and steps.mean() == pytest.approx(0.05, abs=0.005)
    assert len(speeds) == len(p) and speeds.min() >= 1.5 - 1e-9 and speeds.max() <= 5.0 + 1e-9


def test_nearest_index_follows_and_reacquires(path):
    p, _ = path
    n = len(p)
    idx = pp.nearest_index(p, *p[0])
    for i in range(n // 2):                 # half a lap, windowed
        idx = pp.nearest_index(p, *p[i], idx)
    assert idx == n // 2 - 1
    idx = pp.nearest_index(p, *p[0], idx)   # pose reset back to the start
    assert idx == 0


def make(path, **kw):
    p, speeds = path
    return pp.PurePursuit(p, speeds, **kw)


def test_fixed_speed_ignores_confidence(path):
    a = make(path, speed_scale=0.6)
    x, y = a.path[0]
    for c in (None, 0.0, 1.0):
        _, command, _ = a.step(x, y, 0.0, 0.0, c, 0.025)
        assert command == pytest.approx(0.6 * a.speeds[a.index])


def test_adaptive_speed_ramps_up_and_drops_at_once(path):
    a = make(path, speed_scale=0.6, adaptive_speed=True, conf_min_speed=0.8, conf_rise_rate=0.5)
    x, y = a.path[0]
    full = 0.6 * a.speeds[0]
    assert a.step(x, y, 0.0, 0.0, None, 0.025)[1] == pytest.approx(min(0.8, full))
    commands = [a.step(x, y, 0.0, 0.0, 1.0, 0.1)[1] for _ in range(25)]
    assert np.all(np.diff(commands) >= -1e-9)            # never faster than the rise limit
    assert a.confidence == pytest.approx(1.0)            # 2 s at 0.5 / s
    assert commands[0] < full and commands[-1] == pytest.approx(full)
    assert a.step(x, y, 0.0, 0.0, 0.0, 0.025)[1] == pytest.approx(min(0.8, full))  # instant drop


def test_goal_is_ahead_of_the_car(path):
    a = make(path)
    p = a.path
    for i in range(0, len(p), 25):
        heading = np.arctan2(*(p[(i + 5) % len(p)] - p[i])[::-1])
        _, _, goal = a.step(p[i, 0], p[i, 1], heading, 2.0, None, 0.025)
        fwd, _ = pp.to_vehicle_frame(goal, p[i, 0], p[i, 1], heading)
        assert fwd > 0.5
