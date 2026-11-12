#!/usr/bin/env python3
"""
Print the filter's health once a second: means of /pf/stats over that second, the latest
/pf/confidence and the latest /drive speed.

Usage (with the filter running): python3 scripts/monitor.py
Healthy laps: inlier near 1 (sim) or steady (car), confidence near 1, spread a few cm.
"""

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32, Float32MultiArray


def main():
    rclpy.init()
    node = rclpy.create_node('fwc_monitor')
    stats, latest = [], {'confidence': None, 'speed': None}
    node.create_subscription(Float32MultiArray, '/pf/stats',
                             lambda m: stats.append(list(m.data)), 10)
    node.create_subscription(Float32, '/pf/confidence',
                             lambda m: latest.__setitem__('confidence', m.data), 10)
    node.create_subscription(AckermannDriveStamped, '/drive',
                             lambda m: latest.__setitem__('speed', m.drive.speed), 10)

    def show(name, fmt):
        return '-' if latest[name] is None else fmt % latest[name]

    def tick():
        if not stats:
            print('no /pf/stats yet (filter waiting for a 2D Pose Estimate?)')
            return
        inlier, spread, ess = np.mean(stats, axis=0)
        print('%3d updates/s  inlier %.2f  spread %.3f m  ess %.2f  confidence %s  speed %s m/s'
              % (len(stats), inlier, spread, ess, show('confidence', '%.2f'),
                 show('speed', '%.2f')))
        stats.clear()

    node.create_timer(1.0, tick)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
