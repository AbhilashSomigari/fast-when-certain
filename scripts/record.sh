#!/bin/bash
# Record one run into ~/bags/<MMDD_HHMM>_<name>.
# Usage: ./record.sh <name>        e.g. ./record.sh slow_test
# Topics that don't exist yet (EKF, confidence) are picked up if they appear.
set -e
if [ -z "$1" ]; then
  echo "usage: $0 <name>"
  exit 1
fi
mkdir -p ~/bags
ros2 bag record -o ~/bags/"$(date +%m%d_%H%M)_$1" \
  /scan /odom /sensors/imu/raw /tf /tf_static /drive /teleop /initialpose \
  /pf/pose/odom /pf/stats /pf/confidence /odometry/filtered
