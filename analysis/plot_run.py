#!/usr/bin/env python3
"""
Plot one run: the filter's path over the map, and the 3 /pf/stats values over time
(with /pf/confidence on the inlier panel when confidence.csv exists). With offset.csv and
mode.csv (obstacle avoidance), a 4th panel: distance from the raceline, AVOID (red) and FOLLOW
(green) periods shaded, which shows the car going round an obstacle and back onto the line.

Usage: python3 plot_run.py <csv_dir> [map.yaml]
  csv_dir  - output of scripts/bag_to_csv.py (needs pose.csv, stats.csv)
  map.yaml - optional, e.g. fwc_pf/maps/track.yaml, to draw the map underneath
Saves <csv_dir>/run.png.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def read_pgm(path):
    # binary PGM (P5), the format map_saver writes
    with open(path, 'rb') as f:
        data = f.read()
    fields, i = [], 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            i = data.index(b'\n', i)
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    assert fields[0] == b'P5', 'only binary PGM (P5) is supported'
    width, height = int(fields[1]), int(fields[2])
    return np.frombuffer(data, np.uint8, width * height, i + 1).reshape(height, width)


def draw_map(ax, map_yaml):
    with open(map_yaml) as f:
        info = yaml.safe_load(f)
    image = os.path.join(os.path.dirname(map_yaml), info['image'])
    img = read_pgm(image) if image.endswith('.pgm') else plt.imread(image)
    res, (x0, y0) = info['resolution'], info['origin'][:2]
    h, w = img.shape[:2]
    ax.imshow(img, cmap='gray', extent=[x0, x0 + w * res, y0, y0 + h * res])


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    csv_dir = os.path.expanduser(sys.argv[1])
    pose = read_csv(os.path.join(csv_dir, 'pose.csv'))
    stats = read_csv(os.path.join(csv_dir, 'stats.csv'))
    conf_path = os.path.join(csv_dir, 'confidence.csv')
    conf = read_csv(conf_path) if os.path.exists(conf_path) else None

    offset_path = os.path.join(csv_dir, 'offset.csv')
    mode_path = os.path.join(csv_dir, 'mode.csv')
    offset = read_csv(offset_path) if os.path.exists(offset_path) else None
    mode = read_csv(mode_path) if os.path.exists(mode_path) else None
    rows = 4 if offset is not None else 3

    fig = plt.figure(figsize=(13, 2 * rows))
    grid = fig.add_gridspec(rows, 2, width_ratios=[1.1, 1])

    ax = fig.add_subplot(grid[:, 0])
    if len(sys.argv) == 3:
        draw_map(ax, os.path.expanduser(sys.argv[2]))
    ax.plot(pose['x'], pose['y'], lw=1.2, color='tab:blue', label='filter pose')
    ax.plot(pose['x'][0], pose['y'][0], 'o', color='tab:green', label='start')
    pad = 1.0
    ax.set_xlim(pose['x'].min() - pad, pose['x'].max() + pad)
    ax.set_ylim(pose['y'].min() - pad, pose['y'].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper right')
    ax.set_title(os.path.basename(csv_dir.rstrip('/')))

    signals = [('inlier', 'inlier ratio'), ('spread', 'spread (m)'), ('ess', 'ESS / N')]
    for row, (name, label) in enumerate(signals):
        sax = fig.add_subplot(grid[row, 1])
        sax.plot(stats['t'], stats[name], lw=0.8, color='tab:orange', label=label)
        if name == 'inlier' and conf is not None:
            sax.plot(conf['t'], conf['confidence'], lw=0.8, color='tab:purple',
                     label='confidence')
            sax.set_ylim(-0.05, 1.05)
            sax.legend(loc='lower right')
            label = 'inlier / confidence'
        sax.set_ylabel(label)
        sax.grid(alpha=0.3)
        if row < rows - 1:
            sax.tick_params(labelbottom=False)
    if offset is not None:
        sax = fig.add_subplot(grid[3, 1])
        sax.plot(offset['t'], offset['offset'], lw=0.8, color='tab:blue')
        if mode is not None and np.size(mode['t']) > 1:
            t, m = np.atleast_1d(mode['t']), np.atleast_1d(mode['mode'])
            sax.fill_between(t, 0, 1, where=(m > 0.5) & (m < 1.5), step='post', alpha=0.2,
                             color='tab:red', transform=sax.get_xaxis_transform(),
                             label='AVOID')
            if (m > 1.5).any():
                sax.fill_between(t, 0, 1, where=m > 1.5, step='post', alpha=0.2,
                                 color='tab:green', transform=sax.get_xaxis_transform(),
                                 label='FOLLOW')
            sax.legend(loc='upper right')
        sax.set_ylabel('off raceline (m)')
        sax.grid(alpha=0.3)
    sax.set_xlabel('time (s)')

    fig.tight_layout()
    out = os.path.join(csv_dir, 'run.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()
