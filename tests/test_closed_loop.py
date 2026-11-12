"""Closed loop in the 2D simulator: filter + confidence + pure pursuit driving the car.

Each test prints a summary line (run pytest with -s to see them). About 4 minutes in all.
"""

import numpy as np
import pytest

import sim2d


@pytest.fixture(scope='module')
def track():
    return sim2d.World('track')


def after(log, t0):
    return log['t'] >= t0


def lap_times(log):
    ends = log['t'][np.flatnonzero(np.diff(log['lap']) > 0) + 1]
    return np.diff(np.concatenate([[0.0], ends]))


def sideways(log):
    h = log['true'][:, 2]
    d = log['pose'][:, :2] - log['true'][:, :2]
    return np.abs(-np.sin(h) * d[:, 0] + np.cos(h) * d[:, 1])


def test_baseline_laps_class_track(track):
    log = sim2d.run_laps(track, 'track', laps=3)
    print('\n', sim2d.summary(log))
    settled = after(log, 2.0)
    assert log['laps'] == 3
    assert log['err'][settled].mean() < 0.08
    assert log['err'][settled].max() < 0.3
    assert log['offset'].max() < 0.3
    assert log['clearance'].min() > 0.2


def test_baseline_lap_levine():
    log = sim2d.run_laps(sim2d.World('levine'), 'levine', laps=1)
    print('\n', sim2d.summary(log))
    # levine's corridors are longer than the 10 m lidar range, so the pose may slide along
    # them; what steering needs is the sideways error
    assert log['laps'] == 1
    assert sideways(log).max() < 0.2
    assert log['offset'].max() < 0.3
    assert log['clearance'].min() > 0.2


def test_pose_reset_mid_lap(track):
    # a new 2D Pose Estimate 0.4 m / 0.3 rad off, e.g. after a takeover
    log = sim2d.run_laps(track, 'track', laps=2, events=[(5.0, 'reset', (0.4, -0.3, 0.3))])
    print('\n', sim2d.summary(log))
    back = after(log, 7.0)
    assert log['laps'] == 2
    assert log['err'][back].max() < 0.2
    assert log['offset'].max() < 0.4
    assert log['clearance'].min() > 0.2


def test_noise_does_not_lower_confidence(track):
    logs = {k: sim2d.run_laps(track, 'track', laps=1, adaptive_speed=True, noise_speed_gain=k)
            for k in (0.0, 0.5)}
    for k, log in logs.items():
        print('\n k=%.1f' % k, sim2d.summary(log))
    settled = {k: after(log, 3.0) for k, log in logs.items()}
    inlier = {k: logs[k]['inlier'][settled[k]].mean() for k in logs}
    conf = {k: logs[k]['confidence'][settled[k]].mean() for k in logs}
    spread = {k: logs[k]['spread'][settled[k]].mean() for k in logs}
    assert abs(inlier[0.5] - inlier[0.0]) < 0.02
    assert conf[0.0] > 0.9 and conf[0.5] > 0.9
    # the old signal: the injected noise widens the cloud by itself
    assert spread[0.5] > 1.1 * spread[0.0]


def test_kidnap_slows_down_then_recovers(track):
    # the real car jumps 0.8 m and the filter isn't told; on this small track it finds the
    # car again by itself within about a second
    _, ctl = sim2d.experiment_params()
    log = sim2d.run_laps(track, 'track', laps=1, adaptive_speed=True, max_time=10.0,
                         events=[(5.0, 'kidnap', (0.8, 0.0, 0.0))])
    print('\n', sim2d.summary(log))
    t = log['t']
    lost = (t > 5.0) & (t < 5.5)
    assert log['confidence'][(t > 4.0) & (t < 5.0)].min() > 0.9
    assert log['confidence'][lost].min() < 0.3
    assert log['command'][lost].min() < ctl['conf_min_speed'] + 0.4
    assert log['confidence'][t > 8.0].min() > 0.9
    assert log['err'][t > 8.0].max() < 0.2


def test_lidar_occlusion_slows_down(track):
    _, ctl = sim2d.experiment_params()
    log = sim2d.run_laps(track, 'track', laps=1, adaptive_speed=True, max_time=12.0,
                         events=[(5.0, 'occlude', 1.5)])
    print('\n', sim2d.summary(log))
    t = log['t']
    assert log['confidence'][(t > 5.5) & (t < 6.5)].max() < 0.3
    assert log['command'][(t > 5.5) & (t < 6.5)].max() < ctl['conf_min_speed'] + 0.3
    assert log['confidence'][t > 8.0].min() > 0.9
    assert log['clearance'].min() > 0.2


def test_adaptive_speed_costs_nothing_when_sure(track):
    times = {}
    for adaptive in (False, True):
        log = sim2d.run_laps(track, 'track', laps=2, adaptive_speed=adaptive)
        print('\n adaptive=%s' % adaptive, sim2d.summary(log), ' laps', np.round(lap_times(log), 2))
        times[adaptive] = lap_times(log)[1]          # lap 2: past the start ramp
    assert times[True] < 1.02 * times[False]
