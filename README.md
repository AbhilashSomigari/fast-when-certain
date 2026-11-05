# Fast When Certain

Uncertainty-aware speed control for an F1TENTH car. The particle filter reports how sure it is of the car's pose, and the car slows down only when it isn't sure. Course project, Fall 2026. Work in progress.

## Repository layout

```
fwc_pf/          particle filter (ROS 2 package), from f1tenth/particle_filter foxy-devel,
                 plus /pf/stats, /pf/confidence and speed-scaled motion noise
  maps/          track map goes here (track.pgm + track.yaml); levine is the sim map
fwc_controller/  pure pursuit with confidence-aware speed (from my Lab 4 node) and
                 the experiment launch file
  config/        experiment.yaml: all tuned values
  waypoints/     raceline: track.csv (car), levine.csv (sim)
scripts/         car_setup.sh (install + build), record.sh (record a run),
                 bag_to_csv.py (bag -> CSV files), replay.sh (rerun the filter on a bag)
analysis/        plot_run.py (one run on the map), summarize.py (laps, failures, results)
docs/            run_log.md
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
- `ros2 topic echo /pf/stats`: `[spread, ess, match]`, with spread a few cm when localized

The filter publishes TF `map → laser` (upstream behavior). If the scan flickers in RViz, that's this TF, not the estimate. `/pf/pose/odom` is unaffected, and it's what the project uses.

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
python3 analysis/summarize.py --signals fwc_controller/waypoints/track.csv ~/bags/*_csv  # pick the signal
```

**6. Tune the noise gain by replay** (VM, no car): run a recorded bag through the filter again with
a different `noise_speed_gain` k. The bag must contain the `/initialpose` (start recording before the
2D Pose Estimate).

```bash
for k in 0 0.25 0.5 1.0; do ./scripts/replay.sh ~/bags/<bag name> $k; done
python3 analysis/summarize.py fwc_controller/waypoints/track.csv ~/bags/*_replay_k*_csv   # grouped by k
```

Fewer pose jumps (and, on a tie, lower mean spread) is better.

## Confidence signals (`/pf/stats`)

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | spread | Weighted std of particle positions (m). Grows when the filter is unsure. |
| 1 | ess | Effective sample size ÷ number of particles (0–1). Low when few particles explain the scan. |
| 2 | match | Log of the mean unnormalized particle weight. Low when the scan fits the map poorly. |

`/pf/confidence` (0–1) is one of these, chosen by `confidence_signal` and mapped linearly
between `conf_low` and `conf_high` in `fwc_controller/config/experiment.yaml`.
