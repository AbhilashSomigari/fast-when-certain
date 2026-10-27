#!/usr/bin/env python3
"""
Convert one rosbag2 recording into CSV files, one per topic.

Usage: python3 bag_to_csv.py ~/bags/1030_1415_slow_test
Writes ~/bags/1030_1415_slow_test_csv/{pose,stats,drive,...}.csv.
Column t is the bag receive time in seconds since the first message.
"""

import csv
import math
import os
import sys

import rosbag2_py
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
    '/pf/stats': ('stats.csv', ['spread', 'ess', 'match'], lambda m: list(m.data)),
    '/pf/confidence': ('confidence.csv', ['confidence'], lambda m: [m.data]),
    '/drive': ('drive.csv', ['speed', 'steer'], drive_row),
    '/teleop': ('teleop.csv', ['speed', 'steer'], drive_row),
}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bag = os.path.expanduser(sys.argv[1]).rstrip('/')
    out_dir = bag + '_csv'
    os.makedirs(out_dir, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    files, writers, counts = {}, {}, {}
    for topic, (name, columns, _) in TOPICS.items():
        if topic in types:
            files[topic] = open(os.path.join(out_dir, name), 'w', newline='')
            writers[topic] = csv.writer(files[topic])
            writers[topic].writerow(['t'] + columns)
            counts[topic] = 0

    t0 = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
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
