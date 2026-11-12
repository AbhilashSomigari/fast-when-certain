"""
Monte Carlo localization, kept free of ROS so it can be tested on its own.

The math is the f1tenth particle filter's (MIT, Walsh / Zheng et al.): resample, odometry
motion model with Gaussian noise, beam sensor model from a precomputed table, squash factor.
Ray casting is passed in as a function, so the ROS node uses range_libc and the tests use a
numpy ray caster.

Every update (ParticleFilter.update):
  1. resample the particles by their weights
  2. motion model: apply the odometry step in each particle's frame, plus Gaussian noise in
     map x, y and heading (map frame, as upstream), scaled by (1 + noise_speed_gain * |speed|)
     (adaptive noise; gain 0 = fixed noise)
  3. sensor model: ray cast every particle, look up the probability of each measured beam
     in the table, multiply over beams (summed in log space), squash
  4. pose = weighted mean, with a circular mean for the heading
  5. stats [inlier, spread, ess]:
       inlier - fraction of beams whose measured range is within inlier_threshold of the
                range ray cast from the map at the pose (how well the scan fits the map
                there). Only the pose is used, not the particle cloud, so the motion noise
                does not change it. This is the confidence signal.
       spread - weighted std of the particle positions (m)
       ess    - effective sample size / number of particles
"""

import numpy as np

DEFAULTS = {
    'max_particles': 2000,
    'squash_factor': 2.2,
    'max_range': 10.0,              # m
    'z_hit': 0.75,
    'z_short': 0.01,
    'z_max': 0.07,
    'z_rand': 0.12,
    'sigma_hit': 8.0,               # map cells
    'motion_dispersion_x': 0.05,    # m per update
    'motion_dispersion_y': 0.025,
    'motion_dispersion_theta': 0.25,  # rad per update
    'noise_speed_gain': 0.0,        # k, per m/s
    'inlier_threshold': 0.2,        # m
}


def wrap(angle):
    """Angle in [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def odom_delta(prev, pose):
    """Motion from odometry pose prev to pose (x, y, yaw), in prev's frame."""
    c, s = np.cos(prev[2]), np.sin(prev[2])
    dx, dy = pose[0] - prev[0], pose[1] - prev[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - prev[2])])


def sensor_table(max_range_px, z_hit, z_short, z_max, z_rand, sigma_hit):
    """table[r, d] = probability of measuring r cells when the map says d cells.

    Same model as the upstream precompute_sensor_model; each column is normalized.
    """
    n = int(max_range_px) + 1
    r = np.arange(n, dtype=float)[:, None]
    d = np.arange(n, dtype=float)[None, :]
    z = r - d
    prob = z_hit * np.exp(-z * z / (2.0 * sigma_hit ** 2)) / (sigma_hit * np.sqrt(2.0 * np.pi))
    prob = prob + np.where(r < d, 2.0 * z_short * (d - r) / np.maximum(d, 1.0), 0.0)
    prob = prob + np.where(r == int(max_range_px), z_max, 0.0)
    prob = prob + np.where(r < int(max_range_px), z_rand / float(max_range_px), 0.0)
    return prob / prob.sum(axis=0, keepdims=True)


class GridMap:
    """Free space of an occupancy grid. free[row, col], row 0 at the map origin (ROS order)."""

    def __init__(self, free, resolution, origin):
        self.free = np.asarray(free, dtype=bool)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)   # x, y, yaw of cell (0, 0)
        self.free_cells = np.argwhere(self.free)        # (row, col)

    def cells_to_world(self, rows, cols):
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        mx, my = cols * self.resolution, rows * self.resolution
        return (self.origin[0] + c * mx - s * my, self.origin[1] + s * mx + c * my)


class ParticleFilter:
    """Particles, weights and the latest pose and stats. See the module docstring."""

    def __init__(self, grid, raycast, rng=None, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError('unknown parameters: %s' % sorted(unknown))
        self.p = dict(DEFAULTS, **params)
        self.grid = grid
        self.raycast = raycast   # (poses (N, 3), angles (M,)) -> ranges (N, M) in m
        self.rng = np.random.default_rng() if rng is None else rng
        self.n = int(self.p['max_particles'])
        self.max_range_px = int(self.p['max_range'] / grid.resolution)
        table = sensor_table(self.max_range_px, self.p['z_hit'], self.p['z_short'],
                             self.p['z_max'], self.p['z_rand'], self.p['sigma_hit'])
        self.log_table = np.log(table)
        self.particles = None
        self.weights = np.full(self.n, 1.0 / self.n)
        self.pose = None
        self.stats = None
        self.expected_ranges = None   # ray cast at the pose, for the fake scan in RViz

    @property
    def initialized(self):
        return self.particles is not None

    def init_pose(self, x, y, yaw):
        """Particles around a pose (2D Pose Estimate), as upstream: 0.5 m, 0.4 rad."""
        self.particles = np.column_stack([
            x + self.rng.normal(0.0, 0.5, self.n),
            y + self.rng.normal(0.0, 0.5, self.n),
            yaw + self.rng.normal(0.0, 0.4, self.n)])
        self.weights[:] = 1.0 / self.n

    def init_global(self):
        """Particles spread over the free space with random headings (Publish Point)."""
        cells = self.grid.free_cells[self.rng.integers(0, len(self.grid.free_cells), self.n)]
        # anywhere inside each chosen cell
        x, y = self.grid.cells_to_world(cells[:, 0] + self.rng.uniform(0.0, 1.0, self.n),
                                        cells[:, 1] + self.rng.uniform(0.0, 1.0, self.n))
        self.particles = np.column_stack([x, y, self.rng.uniform(-np.pi, np.pi, self.n)])
        self.weights[:] = 1.0 / self.n

    def motion_model(self, action, speed):
        c, s = np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2])
        scale = 1.0 + self.p['noise_speed_gain'] * abs(speed)
        noise = self.rng.normal(0.0, 1.0, (self.n, 3)) * scale * np.array([
            self.p['motion_dispersion_x'], self.p['motion_dispersion_y'],
            self.p['motion_dispersion_theta']])
        self.particles[:, 0] += c * action[0] - s * action[1] + noise[:, 0]
        self.particles[:, 1] += s * action[0] + c * action[1] + noise[:, 1]
        self.particles[:, 2] += action[2] + noise[:, 2]

    def to_cells(self, ranges):
        cells = np.nan_to_num(np.asarray(ranges, dtype=float) / self.grid.resolution,
                              nan=self.max_range_px, posinf=self.max_range_px)
        return np.rint(np.clip(cells, 0, self.max_range_px)).astype(np.intp)

    def sensor_model(self, obs, angles):
        ranges = self.raycast(self.particles, angles)
        log_w = self.log_table[self.to_cells(obs)[None, :], self.to_cells(ranges)].sum(axis=1)
        log_w /= self.p['squash_factor']
        w = np.exp(log_w - log_w.max())
        self.weights = w / w.sum()

    def expected_pose(self):
        x, y = self.weights @ self.particles[:, :2]
        yaw = np.arctan2(self.weights @ np.sin(self.particles[:, 2]),
                         self.weights @ np.cos(self.particles[:, 2]))
        return np.array([x, y, yaw])

    def compute_stats(self, obs, angles):
        self.expected_ranges = self.raycast(self.pose[None, :], angles)[0]
        max_r = self.p['max_range']
        measured = np.clip(np.nan_to_num(np.asarray(obs, dtype=float), nan=max_r, posinf=max_r),
                           0.0, max_r)
        expected = np.clip(self.expected_ranges, 0.0, max_r)
        inlier = float(np.mean(np.abs(measured - expected) < self.p['inlier_threshold']))
        sq = np.sum((self.particles[:, :2] - self.pose[:2]) ** 2, axis=1)
        spread = float(np.sqrt(self.weights @ sq))
        ess = float(1.0 / np.sum(self.weights ** 2) / self.n)
        return [inlier, spread, ess]

    def update(self, action, speed, obs, angles):
        """One MCL step. action = odometry step (forward, left, dyaw) since the last update,
        speed = current speed (m/s), obs = measured ranges (m) at angles (rad, laser frame)."""
        idx = self.rng.choice(self.n, self.n, p=self.weights)
        self.particles = self.particles[idx]
        self.motion_model(action, speed)
        self.sensor_model(obs, angles)
        self.pose = self.expected_pose()
        self.stats = self.compute_stats(obs, angles)
        return self.pose


class Confidence:
    """Map a signal linearly to 0..1 between low (0) and high (1), then smooth it (EMA).

    Starts at 0 (unsure) and after every reset, so the car pulls away slowly.
    """

    def __init__(self, low, high, alpha):
        if high == low:
            raise ValueError('conf_low and conf_high must differ')
        self.low, self.high, self.alpha = float(low), float(high), float(alpha)
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def update(self, signal):
        c = min(max((signal - self.low) / (self.high - self.low), 0.0), 1.0)
        self.value += self.alpha * (c - self.value)
        return self.value
