"""
Pure pursuit math, kept free of ROS so it can be tested on its own.

Path preparation (once, at start-up):
  1. load_waypoints   - read x, y from the CSV: either plain x, y[, ...]
                        rows, or a pose.csv from scripts/bag_to_csv.py
                        (header row, columns t, x, y, ...).
  2. prepare_path     - drop points where the recorded car backed up or
                        jittered (forward_only), keep the first lap only
                        (first_lap), resample at a uniform arc-length
                        spacing, then circular moving-average smoothing.
  3. speed_profile    - per-waypoint speed limited by lateral acceleration
                        (v = sqrt(a_lat / |kappa|)), then a backward pass so
                        the car can brake in time for the next corner.

Every pose update:
  4. nearest_index    - closest waypoint, searched in a window ahead of the
                        previous one (global search on the first call).
  5. lookahead_point  - first point along the path, going forward from the
                        nearest waypoint, at exactly distance L from the car
                        (circle / segment intersection).
  6. to_vehicle_frame + steering_angle - curvature gamma = 2 y / L^2 of the
     arc through the goal point, steering = atan(wheelbase * gamma).
  7. confident_speed + limit_rise - (adaptive configs only) blend the
     profile speed down towards a slow speed as localization confidence
     drops; confidence may fall at once but only rises at a limited rate.
"""

import math

import numpy as np


def load_waypoints(path):
    """Return an (N, 2) array of x, y from a comma-separated file.

    A file with a header row (e.g. pose.csv from bag_to_csv.py) is read by
    its x and y columns; a file without one by its first two columns.
    """
    with open(path) as f:
        header = f.readline().strip().split(',')
    if 'x' in header and 'y' in header:
        data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        return data[:, [header.index('x'), header.index('y')]]
    data = np.loadtxt(path, delimiter=',', ndmin=2)
    return data[:, :2]


def forward_only(points, min_step=0.1):
    """Drop points that do not move the path forward.

    A recorded trajectory backs up and jitters (e.g. while the car sits at
    the start). A point is kept only if it is at least `min_step` from the
    last kept point and ahead of it along the current direction of travel.
    The first direction is taken from the first point 1 m away.
    """
    far = np.hypot(*(points - points[0]).T) >= 1.0
    if not far.any():
        return points
    direction = points[np.argmax(far)] - points[0]
    kept = [points[0]]
    for p in points[1:]:
        step = p - kept[-1]
        dist = np.hypot(*step)
        if dist >= min_step and step @ direction > 0.0:
            direction = step
            kept.append(p)
    return np.array(kept)


def first_lap(points, close_dist=0.5, min_length=5.0):
    """Cut a recording of several laps down to the first one.

    The lap ends at the first point back within `close_dist` of the start
    after at least `min_length` metres of driving. A recording that never
    comes back is returned whole.
    """
    step = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(step)])
    back = (np.hypot(*(points - points[0]).T) < close_dist) & (s > min_length)
    if not back.any():
        return points
    return points[:np.argmax(back)]


def prepare_path(points, spacing=0.05, smoothing_window=9):
    """Clean, resample and smooth a closed recorded path.

    Points that back up are dropped, the rest are resampled at a uniform
    `spacing` (m), then smoothed with a circular moving average over
    `smoothing_window` waypoints.
    """
    points = first_lap(forward_only(points, 2.0 * spacing))
    closed = np.vstack([points, points[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 3)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    path = np.column_stack([np.interp(s_new, s, closed[:, i])
                            for i in range(2)])
    if smoothing_window > 1:
        k = int(smoothing_window)
        kernel = np.ones(k) / k
        padded = np.vstack([path[-k:], path, path[:k]])
        path = np.column_stack([
            np.convolve(padded[:, i], kernel, mode='same')[k:-k]
            for i in range(2)])
    return path


def path_curvature(path, span=10):
    """Signed curvature at each waypoint from points `span` steps apart."""
    a = np.roll(path, span, axis=0)
    c = np.roll(path, -span, axis=0)
    ab = path - a
    bc = c - path
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = (np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*(c - a).T))
    return 2.0 * cross / np.maximum(denom, 1e-9)


def speed_profile(path, max_speed, min_speed, max_lateral_accel, max_decel):
    """Speed at each waypoint of a closed, uniformly spaced path."""
    kappa = np.abs(path_curvature(path))
    v = np.sqrt(max_lateral_accel / np.maximum(kappa, 1e-6))
    v = np.clip(v, min_speed, max_speed)
    ds = np.hypot(*(path[1] - path[0]))
    # Backward pass (twice around the loop so it wraps): v_i^2 <= v_{i+1}^2
    # + 2 a ds, i.e. the car can always brake down to the next speed.
    n = len(v)
    for k in range(2 * n - 1, -1, -1):
        i = k % n
        j = (i + 1) % n
        v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * max_decel * ds))
    return v


def nearest_index(path, x, y, prev=None, window=100):
    """Index of the waypoint closest to (x, y).

    With `prev` given, only waypoints from a few steps behind `prev` to
    `window` steps ahead of it are searched, so the car can never jump to
    a part of the track that is close in space but far along the path.
    """
    n = len(path)
    if prev is None:
        idx = np.arange(n)
    else:
        idx = np.arange(prev - 5, prev + window) % n
    d2 = (path[idx, 0] - x) ** 2 + (path[idx, 1] - y) ** 2
    return int(idx[np.argmin(d2)])


def lookahead_point(path, x, y, start, lookahead):
    """Return the goal point at distance `lookahead` from (x, y).

    The path is walked forward from waypoint `start`. Returns
    (point, index of the segment it lies on).
    """
    n = len(path)
    p = np.array([x, y])
    for k in range(n):
        i = (start + k) % n
        a = path[i]
        b = path[(i + 1) % n]
        if np.hypot(*(b - p)) < lookahead:
            continue
        # Solve |a + t (b - a) - p| = L for t in [0, 1], larger root.
        d = b - a
        f = a - p
        qa = d @ d
        qb = 2.0 * (f @ d)
        qc = f @ f - lookahead ** 2
        disc = max(qb * qb - 4.0 * qa * qc, 0.0)
        t = (-qb + math.sqrt(disc)) / (2.0 * qa) if qa > 0 else 1.0
        return a + min(max(t, 0.0), 1.0) * d, i
    # Car is farther than L from the whole path: head for the next waypoint.
    i = (start + 1) % n
    return path[i].copy(), i


def to_vehicle_frame(point, x, y, yaw):
    """Map-frame point -> (forward, left) in the vehicle frame."""
    dx = point[0] - x
    dy = point[1] - y
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def steering_angle(local_x, local_y, wheelbase, max_steering):
    """Pure pursuit steering for a goal point in the vehicle frame."""
    dist2 = local_x ** 2 + local_y ** 2
    if dist2 < 1e-9:
        return 0.0
    gamma = 2.0 * local_y / dist2
    angle = math.atan(wheelbase * gamma)
    return min(max(angle, -max_steering), max_steering)


def confident_speed(target, confidence, slow_speed):
    """Blend from `slow_speed` (confidence 0) up to `target` (confidence 1).

    Never faster than `target`, so a slow corner stays slow.
    """
    slow = min(slow_speed, target)
    return slow + confidence * (target - slow)


def limit_rise(prev, new, max_rise):
    """Follow a drop at once; let a rise move by at most `max_rise`."""
    return new if new <= prev else min(new, prev + max_rise)


def yaw_from_quaternion(qx, qy, qz, qw):
    """Yaw (rad) of a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)
