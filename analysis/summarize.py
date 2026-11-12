#!/usr/bin/env python3
"""
Summarize runs: laps, failures, lap time, tracking error, marker and closure error.

Usage:
  python3 summarize.py <waypoints.csv> <csv_dir> [<csv_dir> ...]
  python3 summarize.py --signals <waypoints.csv> <csv_dir> [<csv_dir> ...]

  waypoints.csv - the raceline the controller follows (fwc_controller/waypoints/track.csv)
  csv_dir       - output of scripts/bag_to_csv.py. The bag name gives the config and
                  speed: it must contain baseline|noise|speed|both and _s<speed_scale>,
                  e.g. 1112_1530_baseline_s0.5 or 1118_1000_final_both_s0.6.
                  Output of scripts/replay.sh (..._replay_k<k>) is grouped by k instead.

Default mode prints one row per bag and per (config, speed), writes results.csv and
failure_rate.png next to the first csv_dir.
--signals: for every failure, how the 3 /pf/stats values (inlier, spread, ess) behaved in
the seconds before it versus on clean laps, and suggested conf_low / conf_high for the
inlier ratio, the confidence signal. Writes signals.png.

Rules (match docs/run_log.md):
  lap       - one pass of the raceline index from start to end (the raceline starts at
              the start mark); a clean lap with hand driving in it is not counted
  failure   - a lap with a takeover (a /teleop burst that starts while the car moves
              faster than TAKEOVER_SPEED; driving back to the start after the last lap starts
              from standstill, so it does not count) or a filter pose jump > JUMP_M.
              Wall touches are not in the bag: add them from the run log by hand.
  marker    - (optional) a stop of >= STOP_S within MARKER_RADIUS of a marker listed in a
              Floor markers table in docs/run_log.md; none listed = no marker error
  closure   - distance between the filter pose at the first and last stop on the start mark
  spread    - mean particle spread (m) during laps; compares noise gains k in replays
"""

import csv
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fwc_controller'))
from fwc_controller import pure_pursuit as pp  # noqa: E402

CONFIGS = ['baseline', 'noise', 'speed', 'both']
COLORS = {'baseline': '#2a78d6', 'noise': '#eb6834', 'speed': '#1baf7a', 'both': '#eda100'}
MARKERS = {'baseline': 'o', 'noise': 's', 'speed': '^', 'both': 'D'}
SIGNALS = ['inlier', 'spread', 'ess']
WORSE_WHEN_HIGH = {'inlier': False, 'spread': True, 'ess': False}

JUMP_M = 0.5            # pose jump between two filter messages that counts as a failure
TAKEOVER_SPEED = 0.3    # m/s; a /teleop burst starting faster than this is a takeover
TELEOP_GAP_S = 1.0      # /teleop messages further apart than this are separate bursts
MOVING_SPEED = 0.2      # m/s; the first lap starts when the car first moves this fast
STOP_SPEED = 0.05       # m/s
STOP_S = 5.0            # a stop must last this long to be measured
MARKER_RADIUS = 1.0     # m
PRE_FAILURE_S = (2.0, 0.2)  # --signals: window before a failure, seconds before it


def read_csv(csv_dir, name):
    path = os.path.join(csv_dir, name)
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data if data.size > 1 else None


def parse_name(csv_dir):
    name = os.path.basename(csv_dir.rstrip('/'))
    configs = [c for c in CONFIGS if re.search(r'(^|_)%s(_|$)' % c, name.replace('_csv', ''))]
    speed = re.search(r'_s(\d+(?:\.\d+)?)', name)
    config = configs[0] if len(configs) == 1 else '?'
    replay = re.search(r'_replay_k(\d+(?:\.\d+)?)', name)
    if replay:
        config = 'k=' + replay.group(1)
    return name, config, (float(speed.group(1)) if speed else np.nan)


def read_markers():
    """Marker name -> (x, y) from the Floor markers table in docs/run_log.md."""
    path = os.path.join(HERE, '..', 'docs', 'run_log.md')
    markers = {}
    for line in open(path):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 4 and re.fullmatch(r'M\d+', cells[0]):
            try:
                markers[cells[0]] = (float(cells[2]), float(cells[3]))
            except ValueError:
                pass
    return markers


def raceline_index(path, pose):
    """Raceline index and cross-track distance of every pose.

    Tracked forward like the controller, but searched globally again whenever the pose is
    more than MARKER_RADIUS off the line (after a pose jump, or while driving by hand).
    """
    idx = np.zeros(len(pose), dtype=int)
    prev = None
    for k, (x, y) in enumerate(zip(pose['x'], pose['y'])):
        prev = pp.nearest_index(path, x, y, prev, window=100)
        idx[k] = prev
        if np.hypot(path[prev, 0] - x, path[prev, 1] - y) > MARKER_RADIUS:
            prev = None
    xte = np.hypot(path[idx, 0] - pose['x'], path[idx, 1] - pose['y'])
    return idx, xte


def stops(pose):
    """(t_start, t_end, x, y) of every stop longer than STOP_S."""
    still = np.abs(pose['speed']) < STOP_SPEED
    out, k = [], 0
    while k < len(still):
        if not still[k]:
            k += 1
            continue
        j = k
        while j < len(still) and still[j]:
            j += 1
        if pose['t'][j - 1] - pose['t'][k] >= STOP_S:
            out.append((pose['t'][k], pose['t'][j - 1],
                        float(np.median(pose['x'][k:j])), float(np.median(pose['y'][k:j]))))
        k = j
    return out


def failure_events(pose, teleop):
    """Times of takeovers and pose jumps, and of hand driving started from standstill."""
    events, manual = [], []
    step = np.hypot(np.diff(pose['x']), np.diff(pose['y']))
    events += [('jump', t) for t in pose['t'][1:][step > JUMP_M]]
    if teleop is not None:
        t = teleop['t']
        starts = t[np.concatenate([[True], np.diff(t) > TELEOP_GAP_S])]
        for ts in starts:
            # speed just before the takeover, so a slow pull-away by hand does not count
            k = min(np.searchsorted(pose['t'], ts - 0.2), len(pose) - 1)
            if abs(pose['speed'][k]) > TAKEOVER_SPEED:
                events.append(('takeover', ts))
            else:
                manual.append(ts)
    return sorted(events, key=lambda e: e[1]), manual


def laps(path, pose, events, manual):
    """(t_start, t_end, completed, failed) per lap; clean laps with hand driving are dropped."""
    idx, _ = raceline_index(path, pose)
    n = len(path)
    # a wrap (end -> start of the raceline) only counts after passing mid-lap, so pose noise
    # around the start mark does not make extra laps
    wraps, armed = [], False
    for k in range(1, len(idx)):
        armed = armed or 0.4 * n < idx[k] < 0.6 * n
        if armed and idx[k - 1] > 0.8 * n and idx[k] < 0.2 * n:
            wraps.append(k)
            armed = False
    moving = np.nonzero(np.abs(pose['speed']) > MOVING_SPEED)[0]
    if len(moving) == 0:
        return []
    t_wraps = list(pose['t'][wraps])
    # the first lap starts when the car first moves, if that is near the start of the raceline
    first = moving[0]
    bounds = ([pose['t'][first]] if idx[first] < 0.2 * n else []) + t_wraps
    bounds = [b for b in bounds if b >= pose['t'][first]]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        failed = any(a <= t < b for _, t in events)
        if failed or not any(a <= t < b for t in manual):
            out.append((a, b, True, failed))
    # an unfinished last lap only counts if it ended in a failure
    if bounds and any(t >= bounds[-1] for _, t in events):
        out.append((bounds[-1], pose['t'][-1], False, True))
    return out


def summarize_bag(path, csv_dir, markers):
    name, config, speed = parse_name(csv_dir)
    pose = read_csv(csv_dir, 'pose.csv')
    if pose is None:
        print('skip %s: no pose.csv' % name)
        return None
    events, manual = failure_events(pose, read_csv(csv_dir, 'teleop.csv'))
    lap_list = laps(path, pose, events, manual)
    _, xte = raceline_index(path, pose)

    # lap time: completed, clean laps, not the first (standing start)
    times = [b - a for i, (a, b, done, failed) in enumerate(lap_list)
             if done and not failed and i > 0]
    in_laps = np.zeros(len(pose), dtype=bool)
    for a, b, _, _ in lap_list:
        in_laps |= (pose['t'] >= a) & (pose['t'] < b)

    stats = read_csv(csv_dir, 'stats.csv')
    spread = np.nan
    if stats is not None:
        in_laps_stats = np.zeros(len(stats), dtype=bool)
        for a, b, _, _ in lap_list:
            in_laps_stats |= (stats['t'] >= a) & (stats['t'] < b)
        if in_laps_stats.any():
            spread = float(np.mean(stats['spread'][in_laps_stats]))

    stop_list = stops(pose)
    marker_err = []
    for _, _, x, y in stop_list:
        d = {m: np.hypot(x - mx, y - my) for m, (mx, my) in markers.items()}
        if d and min(d.values()) < MARKER_RADIUS:
            marker_err.append(min(d.values()))
    closure = np.nan
    start_stops = [s for s in stop_list
                   if np.hypot(s[2] - pose['x'][0], s[3] - pose['y'][0]) < MARKER_RADIUS]
    if len(start_stops) >= 2:
        (_, _, x0, y0), (_, _, x1, y1) = start_stops[0], start_stops[-1]
        closure = np.hypot(x1 - x0, y1 - y0)

    return {
        'bag': name, 'config': config, 'speed_scale': speed,
        'laps': len(lap_list), 'failed': sum(f for *_, f in lap_list),
        'takeovers': sum(k == 'takeover' for k, _ in events),
        'jumps': sum(k == 'jump' for k, _ in events),
        'lap_time': float(np.median(times)) if times else np.nan,
        'xte_mean': float(np.mean(xte[in_laps])) if in_laps.any() else np.nan,
        'xte_max': float(np.max(xte[in_laps])) if in_laps.any() else np.nan,
        'spread': spread,
        'marker_err': float(np.mean(marker_err)) if marker_err else np.nan,
        'closure': float(closure),
        '_events': events, '_laps': lap_list,
    }


def fmt(v):
    if isinstance(v, float):
        return '-' if np.isnan(v) else '%.3g' % v
    return str(v)


def print_table(rows, cols):
    widths = [max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols]
    print('  '.join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print('  '.join(fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


def results(path, csv_dirs):
    markers = read_markers()
    rows = [r for r in (summarize_bag(path, d, markers) for d in csv_dirs) if r]
    cols = ['bag', 'config', 'speed_scale', 'laps', 'failed', 'takeovers', 'jumps',
            'lap_time', 'xte_mean', 'xte_max', 'spread', 'marker_err', 'closure']
    print('Per bag (failures from the bag only; add wall touches from the run log):')
    print_table(rows, cols)

    # pool bags of the same config and speed
    groups = {}
    for r in rows:
        groups.setdefault((r['config'], r['speed_scale']), []).append(r)
    pooled = []
    for (config, speed), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        n = sum(r['laps'] for r in rs)
        f = sum(r['failed'] for r in rs)
        pooled.append({
            'config': config, 'speed_scale': speed, 'laps': n, 'failed': f,
            'failure_rate': f / n if n else np.nan,
            'lap_time': float(np.nanmedian([r['lap_time'] for r in rs])),
            'xte_mean': float(np.nanmean([r['xte_mean'] for r in rs])),
            'spread': float(np.nanmean([r['spread'] for r in rs])),
            'marker_err': float(np.nanmean([r['marker_err'] for r in rs])),
            'closure': float(np.nanmean([r['closure'] for r in rs])),
        })
    print('\nPer config and speed:')
    out_cols = ['config', 'speed_scale', 'laps', 'failed', 'failure_rate',
                'lap_time', 'xte_mean', 'spread', 'marker_err', 'closure']
    print_table(pooled, out_cols)

    out_dir = os.path.dirname(os.path.abspath(csv_dirs[0].rstrip('/')))
    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(out_cols)
        for p in pooled:
            w.writerow([fmt(p[c]) for c in out_cols])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for config in CONFIGS:
        pts = sorted((p['speed_scale'], p['failure_rate'], p['laps']) for p in pooled
                     if p['config'] == config and not np.isnan(p['speed_scale']) and p['laps'])
        if not pts:
            continue
        s, fr, n = zip(*pts)
        ax.plot(s, fr, color=COLORS[config], lw=2, marker=MARKERS[config], ms=8,
                markeredgecolor='white', markeredgewidth=1.5, label=config)
        ax.annotate(config, (s[-1], fr[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color='#333333')
    ax.set_xlabel('speed_scale')
    ax.set_ylabel('failed laps / laps')
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    ax.set_title('Failure rate vs speed', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'failure_rate.png'), dpi=150)
    print('\nwrote', os.path.join(out_dir, 'results.csv'), 'and failure_rate.png')


def auc(bad, good):
    """Probability that a random pre-failure sample is worse than a random clean one."""
    ranks = np.argsort(np.argsort(np.concatenate([good, bad]))) + 1
    r_bad = ranks[len(good):].sum()
    return (r_bad - len(bad) * (len(bad) + 1) / 2) / (len(bad) * len(good))


def signals(path, csv_dirs):
    markers = read_markers()
    clean = {s: [] for s in SIGNALS}
    pre = {s: [] for s in SIGNALS}
    traces = []  # (signal values vs time relative to failure) per failure
    for d in csv_dirs:
        r = summarize_bag(path, d, markers)
        stats = read_csv(d, 'stats.csv')
        if r is None or stats is None:
            continue
        for a, b, _, failed in r['_laps']:
            sel = (stats['t'] >= a) & (stats['t'] < b)
            if not failed:
                for s in SIGNALS:
                    clean[s].append(stats[s][sel])
        for _, tf in r['_events']:
            sel = (stats['t'] >= tf - PRE_FAILURE_S[0]) & (stats['t'] < tf - PRE_FAILURE_S[1])
            for s in SIGNALS:
                pre[s].append(stats[s][sel])
            win = (stats['t'] >= tf - 5.0) & (stats['t'] <= tf + 1.0)
            traces.append((stats['t'][win] - tf, {s: stats[s][win] for s in SIGNALS}))

    if not traces or not any(len(c) for c in clean['spread']):
        sys.exit('need at least one failure and one clean lap')
    print('%d failures. Pre-failure window: %.1f-%.1f s before each one.\n'
          % (len(traces), PRE_FAILURE_S[0], PRE_FAILURE_S[1]))
    print('signal  separation  lead_time_s  suggested yaml')
    suggested = None
    for s in SIGNALS:
        sign = 1.0 if WORSE_WHEN_HIGH[s] else -1.0
        good = sign * np.concatenate(clean[s])
        bad = sign * np.concatenate(pre[s])
        sep = auc(bad, good)
        alarm = np.percentile(good, 95)          # 5% false alarms on clean laps
        fail = np.median(bad) if bad.size else alarm
        # lead time: how long before each failure the signal was already past the alarm level
        leads = []
        for rel, vals in traces:
            before = rel <= 0
            over = sign * vals[s][before] > alarm
            k = len(over)
            while k > 0 and over[k - 1]:
                k -= 1
            leads.append(-rel[before][k] if k < len(over) else 0.0)
        lo, hi = sorted([sign * alarm, sign * fail])
        print('%-7s %10.2f  %11.2f  conf_low: %.4g, conf_high: %.4g'
              % (s, sep, np.median(leads), lo, hi))
        if s == 'inlier':
            suggested = (lo, hi)
    print('\nSuggested for config/experiment.yaml: conf_low: %.3g, conf_high: %.3g'
          % suggested)
    print('(conf_high = inlier ratio that 95% of clean-lap samples beat, conf_low = median '
          'inlier ratio just before failures)')
    print('Separation 0.5 = no better than chance, 1.0 = perfect. '
          'Prefer a signal with lead time > 0.5 s.')

    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(7, 7), sharex=True)
    for ax, s in zip(axes, SIGNALS):
        for rel, vals in traces:
            ax.plot(rel, vals[s], color='#2a78d6', lw=1, alpha=0.5)
        lo, hi = np.percentile(np.concatenate(clean[s]), [5, 95])
        ax.axhspan(lo, hi, color='#888888', alpha=0.15, lw=0, label='clean laps, 5-95%')
        ax.axvline(0, color='#333333', lw=1)
        ax.set_ylabel(s)
        ax.grid(alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc='upper left')
    axes[0].set_title('Signals around each failure (t = 0)', loc='left')
    axes[-1].set_xlabel('time relative to failure (s)')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(csv_dirs[0].rstrip('/'))), 'signals.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def main():
    warnings.simplefilter('ignore', RuntimeWarning)  # nan-means of empty groups
    args = sys.argv[1:]
    mode = results
    if args and args[0] == '--signals':
        mode, args = signals, args[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    path = pp.prepare_path(pp.load_waypoints(os.path.expanduser(args[0])))
    mode(path, [os.path.expanduser(a) for a in args[1:]])


if __name__ == '__main__':
    main()
