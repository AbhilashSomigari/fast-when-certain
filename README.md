# Fast When Certain

Uncertainty-aware speed control for an F1TENTH car. The particle filter reports how sure it is of the car's pose, and the car slows down only when it isn't sure. Course project, Fall 2026. Work in progress.

## Repository layout

```
fwc_pf/          particle filter (ROS 2 package): the f1tenth/particle_filter math in a
                 ROS-free core (mcl.py) with a thin node, plus /pf/stats, /pf/confidence
                 and speed-scaled motion noise
  maps/          track map goes here (track.pgm + track.yaml); levine is the sim map
fwc_controller/  pure pursuit with confidence-aware speed (from my Lab 4 node) and
                 the experiment launch file
  config/        experiment.yaml: all tuned values
  waypoints/     raceline: track.csv (car), levine.csv (sim)
scripts/         car_setup.sh (install + build), record.sh (record a run), monitor.py
                 (filter health once a second), bag_to_csv.py (bag -> CSV files),
                 replay.sh (rerun the filter on a bag)
analysis/        plot_run.py (one run on the map), summarize.py (laps, failures, results)
docs/            run_log.md
tests/           pytest on a laptop, no ROS: a 2D simulator (sim2d.py) drives the real
                 filter and controller code in closed loop; fake_ros.py runs the two
                 nodes and the launch file's parameter setup against it
```

## Tests (no ROS needed)

```bash
python3 -m venv ~/fwc-venv && ~/fwc-venv/bin/pip install numpy scipy matplotlib pyyaml pytest
~/fwc-venv/bin/python -m pytest tests          # about 4 minutes; -s prints a line per run
```

## Setup on the car (ROS 2 Foxy)

**1. Install and build** (one command; rerun it any time to update to the latest version)

```bash
git clone https://github.com/AbhilashSomigari/fast-when-certain.git ~/f1tenth_ws/src/fast-when-certain
bash ~/f1tenth_ws/src/fast-when-certain/scripts/car_setup.sh      # in the VM: ... car_setup.sh ~/sim_ws
source ~/f1tenth_ws/install/setup.bash
```

It installs the Python/ROS dependencies and range_libc (CPU build; the filter uses `cddt` ray
casting with 2000 particles, so no CUDA is needed), then builds `fwc_pf` and `fwc_controller`.

**2. Map the track**

```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch slam_toolbox online_async_launch.py \
  params_file:=$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml
# drive 2 slow laps with the joystick, end where you started, then:
ros2 run nav2_map_server map_saver_cli -f ~/f1tenth_ws/src/fast-when-certain/fwc_pf/maps/track
```

Stop SLAM, rebuild `fwc_pf` so the map gets installed, then commit and push `fwc_pf/maps/track.*`.
Then drive 2 slow, smooth laps from the start mark while recording (step 4), and copy that
bag's `pose.csv` to `fwc_controller/waypoints/track.csv`. The controller cuts it to one lap
and smooths it.

**3. Localize**

```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch fwc_pf localize_launch.py
rviz2 -d ~/f1tenth_ws/src/fast-when-certain/fwc_pf/rviz/pf.rviz
```

Start a recording first (step 4) if you want to replay it later, then click **2D Pose Estimate**
on the car's real spot, and drive slowly. Checks:

- `ros2 topic hz /pf/pose/odom`: about 30 Hz or more
- `python3 scripts/monitor.py`: inlier ratio steady and high, confidence near 1 when localized

The filter publishes TF `map → pf_laser` (its own frame, so the car's TF tree is untouched) and the scan in that frame on `/pf/viz/scan`: in RViz the scan should sit on the map's walls.

**4. Record, convert, plot**

```bash
./scripts/record.sh slow_test                      # Ctrl+C to stop; bag lands in ~/bags/
python3 scripts/bag_to_csv.py ~/bags/<bag name>    # writes ~/bags/<bag name>_csv/
python3 analysis/plot_run.py ~/bags/<bag name>_csv fwc_pf/maps/track.yaml
```

**5. Run an experiment**

```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch fwc_controller experiment_launch.py config:=baseline speed_scale:=0.4
```

`config` is `baseline`, `noise` (speed-scaled motion noise in the filter), `speed`
(confidence-aware speed and lookahead) or `both`. Set the pose in RViz, then hold RB.
In the sim, add `sim:=true` (levine map and raceline, `/ego_racecar/odom`).
`particles:=1000` overrides the particle count for every config.

Name each recording after its config and speed, e.g. `./scripts/record.sh baseline_s0.4`, then:

```bash
python3 analysis/summarize.py fwc_controller/waypoints/track.csv ~/bags/*_csv            # results
python3 analysis/summarize.py --signals fwc_controller/waypoints/track.csv ~/bags/*_csv  # conf_low/high
```

**6. Tune the noise gain by replay** (VM, no car): run a recorded bag through the filter again with
a different `noise_speed_gain` k. The bag must contain the `/initialpose` (start recording before the
2D Pose Estimate).

```bash
for k in 0 0.25 0.5 1.0; do ./scripts/replay.sh ~/bags/<bag name> $k; done
python3 analysis/summarize.py fwc_controller/waypoints/track.csv ~/bags/*_replay_k*_csv   # grouped by k
```

Fewer pose jumps (and, on a tie, lower mean spread) is better.

## Confidence (`/pf/stats`, `/pf/confidence`)

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | inlier | Fraction of lidar beams within `inlier_threshold` (0.2 m) of the range ray cast from the map at the filter's pose. Low when the scan doesn't fit the map there. |
| 1 | spread | Weighted std of particle positions (m). |
| 2 | ess | Effective sample size ÷ number of particles (0–1). |

`/pf/confidence` (0–1) is the inlier ratio mapped linearly between `conf_low` and `conf_high`
in `fwc_controller/config/experiment.yaml`, then smoothed. It uses the inlier ratio because
it looks only at the pose, not at the particle cloud: the adaptive motion noise widens the
cloud (and the spread) by itself, which would make a spread-based confidence slow the car
down for no reason. Spread and ESS are recorded for comparison.
