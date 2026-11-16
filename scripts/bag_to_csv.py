#!/usr/bin/env python3
"""
Convert one rosbag2 recording into CSV files, one per topic.

Usage: python3 bag_to_csv.py ~/bags/1030_1415_slow_test
Writes ~/bags/1030_1415_slow_test_csv/{pose,stats,drive,...}.csv.
Column t is the bag receive time in seconds since the first message.
Reads the bag's sqlite3 files directly: Foxy has no rosbag2_py.
"""

import csv
import glob
import math
import os
import sqlite3
import sys

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def odom_row(m):
    p = m.pose.pose
    return [p.position.x, p.position.y, yaw(p.orientation), m.twist.twist.linear.x]


def drive_row(m):
    return [m.drive.speed, m.drive.steering_angle]


# topic -> (output file, columns after t, row function)
TOPICS = {
    '/pf/pose/odom': ('pose.csv', ['x', 'y', 'yaw', 'speed'], odom_row),
    '/odom': ('odom.csv', ['x', 'y', 'yaw', 'speed'], odom_row),
    '/odometry/filtered': ('ekf.csv', ['x', 'y', 'yaw', 'speed'], odom_row),
    '/pf/stats': ('stats.csv', ['inlier', 'spread', 'ess'], lambda m: list(m.data)),
    '/pf/confidence': ('confidence.csv', ['confidence'], lambda m: [m.data]),
    '/drive': ('drive.csv', ['speed', 'steer'], drive_row),
    '/teleop': ('teleop.csv', ['speed', 'steer'], drive_row),
}
# scripts/replay.sh records the filter's outputs under /replay/...; write them to the same files
for _topic in ['/pf/pose/odom', '/pf/stats', '/pf/confidence']:
    TOPICS['/replay' + _topic] = TOPICS[_topic]


def read_bag(bag):
    """(topic, type, serialized data, receive time in ns) of the TOPICS messages, in time order."""
    files = sorted(glob.glob(os.path.join(bag, '*.db3')))
    if not files or not os.path.exists(os.path.join(bag, 'metadata.yaml')):
        sys.exit('%s is not a finished bag (no .db3 or metadata.yaml): was the recording '
                 'stopped with a single Ctrl+C?' % bag)
    query = ('SELECT t.name, t.type, m.data, m.timestamp FROM messages m '
             'JOIN topics t ON m.topic_id = t.id WHERE t.name IN (%s)'
             % ','.join('?' * len(TOPICS)))
    rows = []
    for path in files:
        db = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
        rows += db.execute(query, list(TOPICS)).fetchall()
        db.close()
    rows.sort(key=lambda r: r[3])
    return rows


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bag = os.path.expanduser(sys.argv[1]).rstrip('/')
    out_dir = bag + '_csv'
    os.makedirs(out_dir, exist_ok=True)

    messages = read_bag(bag)
    types = {topic: type_name for topic, type_name, _, _ in messages}

    files, writers, counts = {}, {}, {}
    for topic, (name, columns, _) in TOPICS.items():
        if topic in types:
            files[topic] = open(os.path.join(out_dir, name), 'w', newline='')
            writers[topic] = csv.writer(files[topic])
            writers[topic].writerow(['t'] + columns)
            counts[topic] = 0

    t0 = None
    for topic, _, data, stamp in messages:
        if topic not in writers:
            continue
        t0 = stamp if t0 is None else t0
        msg = deserialize_message(data, get_message(types[topic]))
        writers[topic].writerow([round((stamp - t0) * 1e-9, 4)] + TOPICS[topic][2](msg))
        counts[topic] += 1

    for f in files.values():
        f.close()
    for topic, n in counts.items():
        print(f'{TOPICS[topic][0]:16s} {n:7d} rows  ({topic})')
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
