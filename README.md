# Fast When Certain

Uncertainty-aware speed control for an F1TENTH car. The particle filter reports how sure it is of the car's pose, and the car slows down only when it isn't sure. Course project, Fall 2026. Work in progress.

## Repository layout

```
fwc_pf/       particle filter (ROS 2 package), from f1tenth/particle_filter foxy-devel, plus /pf/stats
  maps/       track map goes here (track.pgm + track.yaml); levine is the sim map
scripts/      record.sh (record a run), bag_to_csv.py (bag -> CSV files)
analysis/     plot_run.py (path on map + confidence signals)
docs/         run_log.md
```

## Setup on the car (ROS 2 Foxy)

**1. Dependencies** (once)

```bash
pip3 install cython transforms3d
sudo apt install ros-foxy-tf-transformations
git clone https://github.com/f1tenth/range_libc.git ~/range_libc
cd ~/range_libc/pywrapper && sudo WITH_CUDA=ON python3 setup.py install
```

If the CUDA build fails, install the CPU version (`sudo python3 setup.py install`) and set `range_method: 'cddt'` and `max_particles: 2000` in `fwc_pf/config/localize.yaml`.

**2. Clone and build**

```bash
cd ~/f1tenth_ws/src && git clone https://github.com/AbhilashSomigari/fast-when-certain.git
cd ~/f1tenth_ws && colcon build --packages-select fwc_pf && source install/setup.bash
```

**3. Map the track**

```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch slam_toolbox online_async_launch.py \
  params_file:=$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml
# drive 2 slow laps with the joystick, end where you started, then:
ros2 run nav2_map_server map_saver_cli -f ~/f1tenth_ws/src/fast-when-certain/fwc_pf/maps/track
```

Stop SLAM, rebuild `fwc_pf` so the map gets installed, then commit and push `fwc_pf/maps/track.*`.

**4. Localize**

```bash
ros2 launch f1tenth_stack bringup_launch.py
ros2 launch fwc_pf localize_launch.py
rviz2 -d ~/f1tenth_ws/src/fast-when-certain/fwc_pf/rviz/pf.rviz
```

Click **2D Pose Estimate** on the car's real spot, then drive slowly. Checks:

- `ros2 topic hz /pf/pose/odom`: about 30 Hz or more
- `ros2 topic echo /pf/stats`: `[spread, ess, match]`, with spread a few cm when localized

The filter publishes TF `map → laser` (upstream behavior). If the scan flickers in RViz, that's this TF, not the estimate. `/pf/pose/odom` is unaffected, and it's what the project uses.

**5. Record, convert, plot**

```bash
./scripts/record.sh slow_test                      # Ctrl+C to stop; bag lands in ~/bags/
python3 scripts/bag_to_csv.py ~/bags/<bag name>    # writes ~/bags/<bag name>_csv/
python3 analysis/plot_run.py ~/bags/<bag name>_csv fwc_pf/maps/track.yaml
```

## Confidence signals (`/pf/stats`)

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | spread | Weighted std of particle positions (m). Grows when the filter is unsure. |
| 1 | ess | Effective sample size ÷ number of particles (0–1). Low when few particles explain the scan. |
| 2 | match | Log of the mean unnormalized particle weight. Low when the scan fits the map poorly. |
