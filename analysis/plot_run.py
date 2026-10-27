#!/usr/bin/env python3
"""
Plot one run: the filter's path over the map, and the 3 confidence signals over time.

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

    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])

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

    signals = [('spread', 'spread (m)'), ('ess', 'ESS / N'), ('match', 'match (log)')]
    for row, (name, label) in enumerate(signals):
        sax = fig.add_subplot(grid[row, 1])
        sax.plot(stats['t'], stats[name], lw=0.8, color='tab:orange')
        sax.set_ylabel(label)
        sax.grid(alpha=0.3)
        if row < 2:
            sax.tick_params(labelbottom=False)
    sax.set_xlabel('time (s)')

    fig.tight_layout()
    out = os.path.join(csv_dir, 'run.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()
