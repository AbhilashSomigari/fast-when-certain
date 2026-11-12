"""
Small 2D simulator for testing on the Mac, without ROS.

  World      - a map from fwc_pf/maps, with a numpy ray caster (sphere tracing on a distance
               transform), standing in for range_libc
  run_laps   - closed loop: kinematic car with drifting odometry and a noisy lidar ->
               mcl.ParticleFilter -> mcl.Confidence -> pure_pursuit.PurePursuit -> car,
               wired the way the ROS nodes are. Returns a log of every filter update.

The lidar, the filter and the map share one ray caster, so this checks the logic, not
real-world effects (wheel slip, reflections, a map that doesn't match the room).
"""

import os
import sys

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ('fwc_pf', 'fwc_controller', 'analysis'):
    sys.path.insert(0, os.path.join(REPO, _pkg))

from fwc_pf import mcl  # noqa: E402
from fwc_controller import pure_pursuit as pp  # noqa: E402
from plot_run import read_pgm  # noqa: E402

MAPS = os.path.join(REPO, 'fwc_pf', 'maps')
WAYPOINTS = os.path.join(REPO, 'fwc_controller', 'waypoints')
# gym / Hokuyo: 1080 beams over 270 deg; the filter keeps every 18th (angle_step)
ANGLES = np.linspace(-2.35, 2.35, 1080)[::18]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def experiment_params():
    """Tuned values from config/experiment.yaml and fwc_pf/config/localize.yaml."""
    pf = load_yaml(os.path.join(REPO, 'fwc_pf', 'config', 'localize.yaml'))
    ex = load_yaml(os.path.join(REPO, 'fwc_controller', 'config', 'experiment.yaml'))
    pf_params = dict(pf['particle_filter']['ros__parameters'])
    pf_params.update(ex['particle_filter']['ros__parameters'])
    return pf_params, dict(ex['controller']['ros__parameters'])


class World:
    """Occupancy map + ray caster. Unknown cells count as free, as in range_libc."""

    def __init__(self, name, max_range=10.0):
        info = load_yaml(os.path.join(MAPS, name + '.yaml'))
        img = read_pgm(os.path.join(MAPS, info['image']))[::-1].astype(float)  # row 0 = origin
        occ = (255.0 - img) / 255.0
        if info.get('negate', 0):
            occ = 1.0 - occ
        self.occupied = occ > info['occupied_thresh']
        self.free = occ < info['free_thresh']
        self.resolution = float(info['resolution'])
        self.origin = np.array(info['origin'][:3], dtype=float)
        self.grid = mcl.GridMap(self.free, self.resolution, self.origin)
        self.dist = distance_transform_edt(~self.occupied)  # cells to the nearest wall
        self.max_px = max_range / self.resolution

    def to_cell(self, x, y):
        return (x - self.origin[0]) / self.resolution, (y - self.origin[1]) / self.resolution

    def clearance(self, x, y):
        """Distance (m) from (x, y) to the nearest wall; 0 outside the map."""
        cx, cy = self.to_cell(x, y)
        h, w = self.dist.shape
        if not (0 <= cx < w and 0 <= cy < h):
            return 0.0
        return float(self.dist[int(cy), int(cx)]) * self.resolution

    def raycast(self, poses, angles):
        """Ranges (m), shape (len(poses), len(angles))."""
        poses = np.asarray(poses, dtype=float)
        cx, cy = self.to_cell(poses[:, 0:1], poses[:, 1:2])
        theta = poses[:, 2:3] + np.asarray(angles)[None, :]
        dx, dy = np.cos(theta), np.sin(theta)
        cx = np.broadcast_to(cx, theta.shape)
        cy = np.broadcast_to(cy, theta.shape)
        h, w = self.dist.shape
        t = np.zeros(theta.shape)
        for _ in range(60):
            ix = np.floor(cx + t * dx).astype(np.intp)
            iy = np.floor(cy + t * dy).astype(np.intp)
            inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            d = np.where(inside, self.dist[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)], 0.0)
            step = np.where(d >= 1.0, d - 0.5, 0.0)
            if not step.any():
                break
            t = np.minimum(t + step, self.max_px)
        return t * self.resolution


def load_path(name, controller_params):
    path = pp.prepare_path(pp.load_waypoints(os.path.join(WAYPOINTS, name + '.csv')),
                           controller_params['waypoint_spacing'],
                           controller_params['smoothing_window'])
    speeds = pp.speed_profile(path, controller_params['max_speed'],
                              controller_params['min_speed'],
                              controller_params['max_lateral_accel'],
                              controller_params['max_decel'])
    return path, speeds


def run_laps(world, path_name, laps=3, speed_scale=0.6, adaptive_speed=False,
             noise_speed_gain=0.0, particles=500, init_error=(0.3, -0.2, 0.2),
             events=(), max_time=None, seed=0, sim_dt=0.005, pf_dt=0.025):
    """Drive `laps` laps from the raceline start. Returns a dict of per-update arrays.

    events: (time s, kind, value) - 'kidnap' (dx, dy, dyaw) moves the real car; 'reset'
    gives the filter a new 2D Pose Estimate at the real car's pose + value (dx, dy, dyaw);
    'occlude' (duration s) blocks the lidar's left half (ranges 0.3 m).
    """
    rng = np.random.default_rng(seed)
    pf_params, ctl = experiment_params()
    pf_keep = {k: v for k, v in pf_params.items() if k in mcl.DEFAULTS}
    pf_keep.update(max_particles=particles, noise_speed_gain=noise_speed_gain)
    pf = mcl.ParticleFilter(world.grid, world.raycast, rng=rng, **pf_keep)
    conf = mcl.Confidence(pf_params['conf_low'], pf_params['conf_high'], pf_params['conf_alpha'])
    path, speeds = load_path(path_name, ctl)
    pursuit = pp.PurePursuit(
        path, speeds, ctl['lookahead_distance'], ctl['lookahead_speed_gain'],
        ctl['wheelbase'], ctl['max_steering'], speed_scale, ctl['search_window'],
        adaptive_speed, ctl['conf_min_speed'], ctl['conf_rise_rate'],
        ctl['lookahead_conf_gain'])

    # real car at the raceline start, facing along it
    heading = np.arctan2(*(path[5] - path[0])[::-1])
    car = np.array([path[0, 0], path[0, 1], heading])
    v = 0.0
    odom = car.copy()        # drifting odometry pose (gym/VESC style)
    odom_anchor = None
    pf.init_pose(car[0] + init_error[0], car[1] + init_error[1], car[2] + init_error[2])
    steer = command = 0.0
    events = sorted(events)
    occluded_until = -1.0

    log = {k: [] for k in ('t', 'true', 'pose', 'inlier', 'spread', 'ess', 'confidence',
                           'command', 'speed', 'offset', 'clearance', 'lap')}
    lap, last_idx = 0, 0
    t, next_pf = 0.0, 0.0
    max_time = max_time or laps * len(path) * 0.05 / (0.25 * speed_scale * ctl['max_speed']) + 20
    while lap < laps and t < max_time:
        while events and events[0][0] <= t:
            _, kind, value = events.pop(0)
            if kind == 'kidnap':
                c, s = np.cos(car[2]), np.sin(car[2])
                car = car + [c * value[0] - s * value[1], s * value[0] + c * value[1], value[2]]
            elif kind == 'reset':
                pf.init_pose(car[0] + value[0], car[1] + value[1], car[2] + value[2])
                odom_anchor = None
                conf.reset()
            elif kind == 'occlude':
                occluded_until = t + value

        # car physics: speed follows the command with an acceleration limit
        v += np.clip(command - v, -9.5 * sim_dt, 7.0 * sim_dt)
        yaw_rate = v * np.tan(steer) / ctl['wheelbase']
        car = car + [v * np.cos(car[2]) * sim_dt, v * np.sin(car[2]) * sim_dt, yaw_rate * sim_dt]
        # odometry: 2 % speed error, 3 % yaw rate error, a little noise
        v_o = v * 1.02 + rng.normal(0, 0.02)
        w_o = yaw_rate * 0.97 + rng.normal(0, 0.02)
        odom = odom + [v_o * np.cos(odom[2]) * sim_dt, v_o * np.sin(odom[2]) * sim_dt, w_o * sim_dt]
        t += sim_dt

        if t < next_pf:
            continue
        next_pf += pf_dt
        obs = world.raycast(car[None, :], ANGLES)[0] + rng.normal(0, 0.01, len(ANGLES))
        if t < occluded_until:
            obs[ANGLES > 0] = 0.3
        if odom_anchor is None:      # the filter's first odometry after a (re)set
            odom_anchor = odom.copy()
            continue
        action = mcl.odom_delta(odom_anchor, odom)
        odom_anchor = odom.copy()
        pose = pf.update(action, v_o, obs, ANGLES)
        c = conf.update(pf.stats[0])
        steer, command, _ = pursuit.step(pose[0], pose[1], pose[2], v_o, c, pf_dt)

        idx = pp.nearest_index(path, car[0], car[1])
        if last_idx > 0.8 * len(path) and idx < 0.2 * len(path):
            lap += 1
        last_idx = idx
        log['t'].append(t)
        log['true'].append(car.copy())
        log['pose'].append(pose.copy())
        log['inlier'].append(pf.stats[0])
        log['spread'].append(pf.stats[1])
        log['ess'].append(pf.stats[2])
        log['confidence'].append(c)
        log['command'].append(command)
        log['speed'].append(v)
        log['offset'].append(float(np.hypot(*(path[idx] - car[:2]))))
        log['clearance'].append(world.clearance(car[0], car[1]))
        log['lap'].append(lap)
    out = {k: np.array(val) for k, val in log.items()}
    out['laps'] = lap
    out['err'] = np.hypot(*(out['pose'][:, :2] - out['true'][:, :2]).T)
    out['yaw_err'] = np.abs(mcl.wrap(out['pose'][:, 2] - out['true'][:, 2]))
    return out


def summary(log):
    """One line per run, printed by the tests."""
    return ('laps %d  time %.1f s  pose err mean %.3f max %.3f m  offset max %.2f m  '
            'clearance min %.2f m  inlier mean %.2f  spread mean %.3f  conf mean %.2f'
            % (log['laps'], log['t'][-1], log['err'].mean(), log['err'].max(),
               log['offset'].max(), log['clearance'].min(), log['inlier'].mean(),
               log['spread'].mean(), log['confidence'].mean()))
