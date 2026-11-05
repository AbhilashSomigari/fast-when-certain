#!/bin/bash
# Replay a recorded run through the particle filter with motion-noise gain k (offline, in the VM).
# Usage: ./replay.sh <bag dir> <k> [extra launch args, e.g. sim:=true]
# Writes <bag dir>_replay_k<k>/ and its CSVs in <bag dir>_replay_k<k>_csv/. The filter starts
# from the /initialpose inside the bag, so the recording must start before the 2D Pose Estimate,
# and the map must be the one the bag was recorded on.
set -e
if [ $# -lt 2 ]; then
  echo "usage: $0 <bag dir> <k> [launch args]"
  exit 1
fi
BAG=$(realpath "$1")
K=$2
shift 2
OUT=${BAG}_replay_k$K
HERE=$(dirname "$(realpath "$0")")
rm -rf "$OUT" "${OUT}_csv"

ros2 launch fwc_controller experiment_launch.py config:=noise replay:=true \
  noise_speed_gain:="$K" "$@" &
LAUNCH=$!
trap 'kill -INT $LAUNCH 2>/dev/null || true' EXIT

# the filter creates its publishers once the map is loaded and its tables are built
for i in $(seq 60); do
  ros2 topic list | grep -q '^/replay/pf/pose/odom$' && break
  sleep 1
done
ros2 topic list | grep -q '^/replay/pf/pose/odom$' || { echo 'filter did not start'; exit 1; }

ros2 bag record -o "$OUT" /replay/pf/pose/odom /replay/pf/stats /replay/pf/confidence &
RECORD=$!
sleep 2
ros2 bag play "$BAG"
sleep 2
kill -INT $RECORD
wait $RECORD || true
python3 "$HERE/bag_to_csv.py" "$OUT"
