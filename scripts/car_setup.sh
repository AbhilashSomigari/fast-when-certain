#!/bin/bash
# One-command setup on the car (or the VM): dependencies, range_libc (CPU version), this repo,
# then build. Safe to rerun: steps that are already done are skipped, and an existing clone is
# updated to the latest pushed version.
# Usage: bash car_setup.sh [workspace]       default ~/f1tenth_ws (in the VM: ~/sim_ws)
set -e
WS=${1:-$HOME/f1tenth_ws}
REPO=$WS/src/fast-when-certain
URL=https://github.com/AbhilashSomigari/fast-when-certain.git
source /opt/ros/foxy/setup.bash

echo '== 1/4 Python and ROS dependencies'
python3 -c 'import Cython, transforms3d' 2>/dev/null || pip3 install --user cython transforms3d
python3 -c 'import tf_transformations' 2>/dev/null || sudo apt-get install -y ros-foxy-tf-transformations

echo '== 2/4 range_libc (CPU)'
if ! (cd / && python3 -c 'import range_libc') 2>/dev/null; then
  [ -d ~/range_libc ] || git clone https://github.com/f1tenth/range_libc.git ~/range_libc
  (cd ~/range_libc/pywrapper && sudo python3 setup.py install)
fi

echo '== 3/4 fast-when-certain'
mkdir -p "$WS/src"
if [ ! -d "$REPO/.git" ]; then
  git clone "$URL" "$REPO"
  cd "$REPO"
else
  cd "$REPO"
  git fetch origin
  # The map and raceline are made on the car, then pushed from the Mac. Drop the car's local
  # copies only once GitHub has them, so the pull can bring them back.
  for f in fwc_pf/maps/track.pgm fwc_pf/maps/track.yaml fwc_controller/waypoints/track.csv; do
    if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 \
        && git cat-file -e "origin/main:$f" 2>/dev/null; then
      rm "$f"
    fi
  done
  if ! git diff --quiet; then
    echo 'Local edits found; saving them with git stash (see: git stash list)'
    git stash
  fi
  git pull --ff-only origin main
fi
git config user.email 197832085+AbhilashSomigari@users.noreply.github.com
echo "Code version: $(git describe --tags --always)"

echo '== 4/4 build'
cd "$WS"
colcon build --packages-select fwc_pf fwc_controller

[ -e "$REPO/fwc_pf/maps/track.yaml" ] || echo 'No track map yet: map the track next.'
[ -e "$REPO/fwc_controller/waypoints/track.csv" ] || echo 'No raceline yet: record it after mapping.'
echo "Done. Now run: source $WS/install/setup.bash"
