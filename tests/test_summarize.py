"""analysis/summarize.py on a synthetic run: 3 laps of the class raceline, a takeover in lap 2."""

import os
import re

import numpy as np
import pytest

import sim2d
import summarize

RATE = 40.0


def write_csv(path, columns, rows):
    with open(path, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in rows:
            f.write(','.join('%.6g' % v for v in row) + '\n')


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """A bag's CSVs: 6 s on the start mark, 3 laps at 2 m/s, 6 s on the start mark."""
    _, ctl = sim2d.experiment_params()
    path, _ = sim2d.load_path('track', ctl)
    n = len(path)
    rng = np.random.default_rng(0)
    still = int(6 * RATE)
    idx = np.concatenate([np.zeros(still, int), np.arange(3 * n) % n, np.zeros(still, int)])
    speed = np.concatenate([np.zeros(still), np.full(3 * n, 2.0), np.zeros(still)])
    t = np.arange(len(idx)) / RATE
    xy = path[idx] + rng.normal(0, 0.01, (len(idx), 2))
    t_fail = t[still + n + n // 2]                       # mid lap 2

    out = tmp_path_factory.mktemp('bags') / '1112_1530_baseline_s0.5_csv'
    os.makedirs(out)
    write_csv(out / 'pose.csv', ['t', 'x', 'y', 'yaw', 'speed'],
              np.column_stack([t, xy, np.zeros(len(t)), speed]))
    inlier = 0.97 + rng.normal(0, 0.01, len(t))
    spread = 0.07 + rng.normal(0, 0.005, len(t))
    before = (t > t_fail - 1.5) & (t <= t_fail)
    inlier[before] = np.linspace(0.95, 0.55, before.sum())
    spread[before] = np.linspace(0.08, 0.25, before.sum())
    write_csv(out / 'stats.csv', ['t', 'inlier', 'spread', 'ess'],
              np.column_stack([t, inlier, spread, rng.uniform(0.05, 0.15, len(t))]))
    t_teleop = np.arange(t_fail, t_fail + 2.0, 0.05)
    write_csv(out / 'teleop.csv', ['t', 'speed', 'steer'],
              np.column_stack([t_teleop, np.full(len(t_teleop), 1.0), np.zeros(len(t_teleop))]))
    return path, str(out), n


def test_laps_failures_and_closure(run):
    path, csv_dir, n = run
    r = summarize.summarize_bag(path, csv_dir, {})
    assert (r['config'], r['speed_scale']) == ('baseline', 0.5)
    assert (r['laps'], r['failed'], r['takeovers'], r['jumps']) == (3, 1, 1, 0)
    assert r['lap_time'] == pytest.approx(n / RATE, abs=0.1)   # lap 3, clean
    assert r['closure'] < 0.05
    assert r['xte_max'] < 0.1


def test_signals_suggest_inlier_thresholds(run, capsys):
    path, csv_dir, _ = run
    summarize.signals(path, [csv_dir])
    text = capsys.readouterr().out
    low, high = map(float, re.search(r'conf_low: ([\d.]+), conf_high: ([\d.]+)', text).groups())
    assert 0.5 < low < high < 0.97
    # the drop starts 1.5 s before the failure, the window 2.0 s: about 1/4 of it is normal
    sep = float(re.search(r'^inlier\s+([\d.]+)', text, re.M).group(1))
    assert 0.75 < sep < 0.95
    assert os.path.exists(os.path.join(os.path.dirname(csv_dir), 'signals.png'))
