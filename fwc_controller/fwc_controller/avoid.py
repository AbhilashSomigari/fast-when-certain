"""
When to leave the raceline and which way round, kept free of ROS so it can be tested.

The controller has two modes:
  TRACK - pure pursuit on the raceline (pure_pursuit.PurePursuit), the default.
  AVOID - follow the gap on the full scan (gap_follow.GapFollow).

On every scan, Driver.step:
  1. scan_points      - every beam as a point in the map, from the filter's (laser) pose.
  2. WallMask         - drop points within wall_margin of a wall in the map: what is left is
                        something the map doesn't have (a box, another car, a person).
  3. corridor_mask    - of those, the points on the raceline ahead: within corridor_half_width
                        of the line, from the car's waypoint to trigger_dist + trigger_time x
                        speed along it.
  4. TRACK -> AVOID   - at least min_points of them in enter_scans scans in a row. The side to
                        pass on is picked from the map: the outside if the raceline bends just
                        before the obstacle and the outside has room, else the side with more
                        room to the walls (track_widths); again for each new obstacle.
     AVOID           - gap follow picks the clear arc closest to pure pursuit's steering
                        towards its goal point, on the chosen side of the obstacle.
     AVOID -> TRACK   - fewer than min_points from exit_behind behind the car to the same
                        distance ahead, for exit_time, counting the obstacle points seen in the
                        last memory_time too (a box beside or behind the car is out of the
                        lidar's view). Pure pursuit then steers back onto the line.

A slower car (the other car): the off-map points are grouped into clusters (cluster_dist)
and each cluster is followed from scan to scan; one that moves faster than moving_speed is the
other car. Its points are not remembered (no trail behind it); instead they are predicted
along its velocity for predict_time and treated as obstacles too.
  TRACK -> FOLLOW    - the other car blocks the raceline ahead and the room beside it on the
                        passing side is under pass_room: pure pursuit keeps steering, the speed
                        holds follow_gap behind it (FOLLOW -> AVOID once there is room, ->
                        TRACK once it is gone from the corridor).
  TRACK -> AVOID     - with room: overtake, keeping moving_margin (not gap follow's
                        path_margin) from it, and back to TRACK only once it is
                        exit_behind_moving behind.
In every mode, a time-to-collision brake: speed 0 while the car's body would reach a scan
point along the arc it is steering in under ttc_brake s at the current speed (Driver.braking).

If the pose is wrong, walls look like obstacles and the car switches to AVOID; gap follow
steers on the scan, so that is safe.
"""

import math

import numpy as np

from fwc_controller.pure_pursuit import path_curvature

TRACK, AVOID, FOLLOW = 0, 1, 2

DEFAULTS = {
    'wall_margin': 0.25,          # m: points this close to a map wall are the wall
    'corridor_half_width': 0.3,   # m: half width of the raceline corridor
    'trigger_dist': 1.8,          # m ahead along the raceline ...
    'trigger_time': 0.3,          # ... + this many seconds at the current speed
    'exit_behind': 0.5,           # m behind the car that must be clear too, to go back
    'exit_time': 0.3,             # s it must stay clear
    'memory_time': 1.5,           # s: obstacle points are kept this long after they were last
                                  # seen (a box beside or behind the car is out of the lidar's view)
    'new_obstacle': 1.0,          # m along the raceline beyond the last one: choose the side again
    'curve_radius': 2.0,          # m: an obstacle within curve_before after a bend tighter
    'curve_before': 1.5,          # than this is passed on the outside if there is outside_room
    'outside_room': 0.25,         # m beside the obstacle to the walls grown by wall_margin
    'min_points': 8,
    'enter_scans': 2,
    'avoid_max_speed': 2.0,       # m/s: speed cap while avoiding
    # the other car
    'cluster_dist': 0.3,          # m: off-map points closer than this (beam to beam) are one object
    'moving_speed': 0.4,          # m/s: an object faster than this is the other car
    'predict_time': 0.75,         # s: its points are also placed where it will be this soon
    'pass_room': 0.3,             # m beside it on the passing side (to the grown walls) to overtake
    'moving_margin': 0.12,        # m kept from it while passing (each side of our car)
    'exit_behind_moving': 1.0,    # m behind us it has to be before going back to TRACK
    'room_scans': 10,             # scans in a row with / without room before AVOID <-> FOLLOW
    'follow_gap': 1.0,            # m kept behind it in FOLLOW ...
    'follow_gain': 1.0,           # ... speed = its speed + gain x (gap - follow_gap)
    'ttc_brake': 0.35,            # s: brake when something straight ahead is closer than this
}


class WallMask:
    """Map walls, and cells within `margin` of them. occupied[row, col], row 0 at the map origin
    (ROS order); origin = x, y, yaw of cell (0, 0)."""

    def __init__(self, occupied, resolution, origin, margin):
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.occupied = np.asarray(occupied, dtype=bool)
        k = int(math.ceil(margin / self.resolution))
        near = self.occupied
        for axis in (0, 1):                     # square dilation, one axis at a time
            grown = near.copy()
            for shift in range(1, k + 1):
                for s in (shift, -shift):
                    rolled = np.roll(near, s, axis=axis)
                    edge = [slice(None)] * 2
                    edge[axis] = slice(0, s) if s > 0 else slice(s, None)
                    rolled[tuple(edge)] = False     # no wrap-around
                    grown |= rolled
            near = grown
        self.near = near

    def cells(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        dx, dy = points[:, 0] - self.origin[0], points[:, 1] - self.origin[1]
        col = np.floor((c * dx + s * dy) / self.resolution).astype(np.intp)
        row = np.floor((-s * dx + c * dy) / self.resolution).astype(np.intp)
        h, w = self.near.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        return row, col, inside

    def near_wall(self, points):
        """True for points near a wall or outside the map."""
        row, col, inside = self.cells(points)
        out = np.ones(len(row), dtype=bool)
        out[inside] = self.near[row[inside], col[inside]]
        return out

    def wall(self, points):
        """True for points on a wall or outside the map."""
        row, col, inside = self.cells(points)
        out = np.ones(len(row), dtype=bool)
        out[inside] = self.occupied[row[inside], col[inside]]
        return out


def occupied_from_grid(data, width, height, threshold=65):
    """Occupied cells of a nav_msgs/OccupancyGrid's data (row-major, -1 = unknown)."""
    return np.asarray(data).reshape(height, width) >= threshold


def path_normals(path):
    """Unit normals pointing to the left of the closed path's direction."""
    d = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    return np.column_stack([-d[:, 1], d[:, 0]])


def track_widths(path, mask, max_width=3.0, behind=0.6, ahead=1.5):
    """Room (m) from each waypoint to the walls on its left and on its right: the distance to
    the first cell within wall_margin of a wall, the smallest from `behind` m before the
    waypoint to `ahead` m after it (the car stays on that side for a while after passing, and
    a row of pillars has gaps a single ray would slip through)."""
    normals = path_normals(path)
    steps = np.arange(1, int(max_width / mask.resolution) + 1) * mask.resolution
    spacing = float(np.mean(np.hypot(*np.diff(path, axis=0).T)))
    shifts = range(-int(round(ahead / spacing)), int(round(behind / spacing)) + 1)
    widths = []
    for sign in (1.0, -1.0):
        pts = path[:, None, :] + sign * steps[None, :, None] * normals[:, None, :]
        hit = mask.near_wall(pts.reshape(-1, 2)).reshape(len(path), len(steps))
        first = steps[np.where(hit.any(axis=1), hit.argmax(axis=1), len(steps) - 1)]
        # np.roll(first, i)[j] = first[j - i]: i < 0 looks ahead, i > 0 behind
        widths.append(np.min([np.roll(first, i) for i in shifts], axis=0))
    return widths[0], widths[1]


def scan_points(ranges, angle_min, inc, pose, max_range=10.0, min_range=0.05):
    """Map-frame points of the valid beams of a scan taken at `pose` (x, y, yaw of the laser)."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + np.arange(r.size) * inc + pose[2]
    ok = np.isfinite(r) & (r > min_range) & (r < max_range)
    return np.column_stack([pose[0] + r[ok] * np.cos(a[ok]), pose[1] + r[ok] * np.sin(a[ok])])


def corridor_mask(points, path, index, s_from, s_to, half_width):
    """Which points lie within half_width of the closed path between arc lengths s_from and
    s_to (m, may be negative) from waypoint `index`. Also returns each point's waypoint."""
    n = len(path)
    if len(points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.intp)
    spacing = float(np.mean(np.hypot(*np.diff(path, axis=0).T)))
    pad = half_width + spacing
    ks = np.arange(int(math.floor((s_from - pad) / spacing)),
                   int(math.ceil((s_to + pad) / spacing)) + 1)
    window = path[(index + ks) % n]
    d2 = ((points[:, None, :] - window[None, :, :]) ** 2).sum(axis=2)
    j = np.argmin(d2, axis=1)
    lateral = np.sqrt(d2[np.arange(len(points)), j])
    s = ks[j] * spacing
    inside = (lateral <= half_width) & (s >= s_from) & (s <= s_to)
    return inside, (index + ks[j]) % n


class Driver:
    """The TRACK / AVOID mode switch. Parameters: DEFAULTS."""

    def __init__(self, path, wall_mask, gap, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError('unknown avoid parameters: ' + ', '.join(sorted(unknown)))
        self.p = dict(DEFAULTS, **params)
        self.path = path
        self.normals = path_normals(path)
        self.spacing = float(np.mean(np.hypot(*np.diff(path, axis=0).T)))
        self.mask = wall_mask
        self.left_width, self.right_width = track_widths(path, wall_mask)
        self.curvature = path_curvature(path)
        self.gap = gap
        self.mode = TRACK
        self.side = 1                 # +1 / -1: pass the obstacle on the raceline's left / right
        self.side_wp = 0              # waypoint of the obstacle the side was chosen for
        self.streak = 0
        self.clear_since = None
        self.switches = 0
        self.memory = {}              # 2 cm cell -> (x, y, time last seen)
        self.tracks = []              # objects seen last scan: centroid, velocity, age
        self.last_t = None
        self.moving = np.zeros((0, 2))    # the other car's points now
        self.predicted = np.zeros((0, 2))  # ... and where they will be
        self.moving_velocity = np.zeros(2)
        self.braking = False
        self.follow_speed = None
        self.room_streak = 0              # scans in a row with (> 0) or without (< 0) room

    def obstacles(self, ranges, angle_min, inc, pose):
        points = scan_points(ranges, angle_min, inc, pose)
        return points[~self.mask.near_wall(points)]

    def remember(self, t, points):
        """Add the points seen now, forget the ones not seen for memory_time; return all."""
        for x, y in points:     # 2 cm cells: a 0.3 m box face still gives 15 of them
            self.memory[(int(math.floor(x / 0.02)), int(math.floor(y / 0.02)))] = (x, y, t)
        old = t - self.p['memory_time']
        self.memory = {k: v for k, v in self.memory.items() if v[2] >= old}
        if not self.memory:
            return np.zeros((0, 2))
        return np.array([v[:2] for v in self.memory.values()])

    def split_moving(self, t, points):
        """(static points, moving points, velocity of the moving ones) from this scan's off-map
        points, which are in beam order: an object = a run of points closer than cluster_dist
        to the previous one; each is matched to the nearest object of the last scan (within
        0.5 m) and its velocity smoothed over scans."""
        p = self.p
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if len(points) == 0:
            self.tracks = []
            return points, np.zeros((0, 2)), np.zeros(2)
        cuts = np.flatnonzero(np.hypot(*np.diff(points, axis=0).T) > p['cluster_dist']) + 1
        groups = np.split(points, cuts)
        tracks, static, moving, velocity = [], [], [], np.zeros(2)
        for g in groups:
            c = g.mean(axis=0)
            v, age = np.zeros(2), 0
            if self.tracks and dt > 0.0:
                prev = min(self.tracks, key=lambda tr: np.hypot(*(tr[0] - c)))
                if np.hypot(*(prev[0] - c)) < 0.5:
                    v = 0.8 * prev[1] + 0.2 * (c - prev[0]) / dt
                    age = prev[2] + 1
            tracks.append((c, v, age))
            if age >= 5 and np.hypot(*v) > p['moving_speed'] and len(g) >= 5:
                moving.append(g)
                velocity = v
            else:
                static.append(g)
        self.tracks = tracks
        static = np.vstack(static) if static else np.zeros((0, 2))
        moving = np.vstack(moving) if moving else np.zeros((0, 2))
        return static, moving, velocity

    def ttc(self, ranges, angle_min, inc, speed, steering):
        """Time (s) until the car's body reaches a scan point along the arc it is steering now,
        at the current speed (the emergency-brake lab's time to collision, along the arc so a
        wall straight ahead in a corner doesn't count)."""
        if speed < 0.1:
            return math.inf
        return self.gap.clear_along(ranges, angle_min, inc, steering) / speed

    def room_on_side(self, points, waypoints, side):
        """Room (m, to the grown walls) beside the points on the given side of the raceline."""
        lateral = np.einsum('ij,ij->i', points - self.path[waypoints], self.normals[waypoints])
        if side > 0:
            return float(np.min(self.left_width[waypoints] - lateral))
        return float(np.min(self.right_width[waypoints] + lateral))

    def choose_side(self, points, waypoints, index):
        """(+1 to pass on the raceline's left or -1 on its right, the obstacle's first waypoint
        after `index`): the side of the obstacle points with more room to the walls."""
        lateral = np.einsum('ij,ij->i', points - self.path[waypoints], self.normals[waypoints])
        left = np.min(self.left_width[waypoints] - lateral)
        right = np.min(self.right_width[waypoints] + lateral)
        n = len(self.path)
        first = int(waypoints[np.argmin((waypoints - index) % n)])
        # right after a bend, the inside needs a tighter turn than the car can make while it
        # is still turning: take the outside if it has room
        bend = self.curvature[(first - np.arange(int(self.p['curve_before'] / self.spacing)))
                              % n].mean()
        if abs(bend) > 1.0 / self.p['curve_radius']:
            outside = -1 if bend > 0 else 1          # left bend: the outside is the right
            if (right if outside < 0 else left) >= self.p['outside_room']:
                return outside, first
        return (1 if left >= right else -1), first

    def step(self, t, pose, index, speed, ranges, angle_min, inc, steering=0.0, goal=None):
        """One scan. pose = laser pose (x, y, yaw), index = pure pursuit's current waypoint,
        speed = current speed, steering = the last steering command, goal = pure pursuit's
        goal point (map frame).

        Returns (mode, command, obstacle points): command = None in TRACK, (steering, speed)
        in AVOID, (None, speed) in FOLLOW (pure pursuit steers, at most this speed). Sets
        self.braking (time-to-collision brake, every mode).
        """
        p = self.p
        n = len(self.path)
        obs = self.obstacles(ranges, angle_min, inc, pose)
        static, moving, velocity = self.split_moving(t, obs)
        self.moving, self.moving_velocity = moving, velocity
        steps = np.arange(1, 4) * p['predict_time'] / 3.0
        self.predicted = (np.vstack([moving + k * velocity for k in steps])
                          if len(moving) else np.zeros((0, 2)))
        dynamic = np.vstack([moving, self.predicted])
        known = self.remember(t, static)
        self.braking = self.ttc(ranges, angle_min, inc, speed, steering) < p['ttc_brake']
        ahead = p['trigger_dist'] + p['trigger_time'] * abs(speed)
        half = p['corridor_half_width']

        # the other car on the raceline ahead: can we pass it here?
        car_in, car_wp = corridor_mask(moving, self.path, index, 0.0, ahead + 1.0, half)
        car_ahead = car_in.sum() >= p['min_points']
        passable = False
        if car_ahead:
            side, _ = self.choose_side(moving[car_in], car_wp[car_in], index)
            passable = self.room_on_side(moving[car_in], car_wp[car_in], side) >= p['pass_room']
            gap = float(((car_wp[car_in] - index) % n).min() * self.spacing)
        # FOLLOW <-> AVOID only after the room has been there (or not) for room_scans in a row
        self.room_streak = (max(self.room_streak, 0) + 1 if passable
                            else min(self.room_streak, 0) - 1)
        room_now = self.room_streak >= p['room_scans']
        no_room_now = self.room_streak <= -p['room_scans']

        if self.mode == TRACK:
            inside, wp = corridor_mask(np.vstack([static, moving]), self.path, index, 0.0,
                                       ahead, half)
            self.streak = self.streak + 1 if inside.sum() >= p['min_points'] else 0
            if self.streak >= p['enter_scans']:
                self.clear_since = None
                self.switches += 1
                if car_ahead and not passable:
                    self.mode = FOLLOW
                else:
                    self.mode = AVOID
                    pts = np.vstack([static, moving])
                    self.side, self.side_wp = self.choose_side(pts[inside], wp[inside], index)
                    self.gap.reset(steering)
        elif self.mode == FOLLOW:
            if car_ahead and room_now:
                self.mode, self.switches = AVOID, self.switches + 1
                self.side, self.side_wp = self.choose_side(moving[car_in], car_wp[car_in],
                                                           index)
                self.gap.reset(steering)
            elif not car_ahead:
                self.clear_since = t if self.clear_since is None else self.clear_since
                if t - self.clear_since >= p['exit_time']:
                    self.mode, self.streak, self.switches = TRACK, 0, self.switches + 1
            else:
                self.clear_since = None
        else:
            # AVOID: back to TRACK once the corridor is clear, from exit_behind behind us for
            # what stands still and from exit_behind_moving for the other car
            still, _ = corridor_mask(known, self.path, index, -p['exit_behind'], ahead, half)
            other, _ = corridor_mask(dynamic, self.path, index, -p['exit_behind_moving'], ahead,
                                     half)
            if still.sum() + other.sum() >= p['min_points']:
                self.clear_since = None
            elif self.clear_since is None:
                self.clear_since = t
            if self.clear_since is not None and t - self.clear_since >= p['exit_time']:
                self.mode, self.streak, self.switches = TRACK, 0, self.switches + 1
            elif car_ahead and no_room_now and gap > 0.5:
                # not alongside it yet and no room any more (e.g. a narrow part): wait behind
                self.mode, self.switches = FOLLOW, self.switches + 1

        self.follow_speed = None
        if self.mode == TRACK:
            return TRACK, None, obs
        if self.mode == FOLLOW:
            if car_ahead:
                tangent = self.path[(car_wp[car_in][0] + 1) % n] - self.path[car_wp[car_in][0]]
                tangent = tangent / max(np.hypot(*tangent), 1e-9)
                along = float(max(velocity @ tangent, 0.0))
                self.follow_speed = max(0.0, along + p['follow_gain'] * (gap - p['follow_gap']))
            else:
                self.follow_speed = abs(speed)
            return FOLLOW, (None, self.follow_speed), obs
        # a new obstacle further along (e.g. the next cone while still avoiding this one):
        # choose the side again for it
        ahead_in, ahead_wp = corridor_mask(obs, self.path, index, 0.0, ahead, half)
        if ahead_in.sum() >= p['min_points']:
            first = ahead_wp[ahead_in][np.argmin((ahead_wp[ahead_in] - index) % n)]
            if (first - self.side_wp) % n > p['new_obstacle'] / self.spacing and \
                    (first - self.side_wp) % n < n / 2:
                self.side, self.side_wp = self.choose_side(obs[ahead_in], ahead_wp[ahead_in],
                                                           index)
        everything = np.vstack([known, dynamic])
        inside, _ = corridor_mask(everything, self.path, index, -p['exit_behind'], ahead, half)
        return AVOID, self.avoid_command(pose, speed, ranges, angle_min, inc,
                                         everything[inside], goal, dynamic), obs

    def avoid_command(self, pose, speed, ranges, angle_min, inc, blocking, goal,
                      dynamic=np.zeros((0, 2))):
        """Gap follow, aimed at the goal point, on the chosen side of the blocking points; the
        other car's predicted points count as obstacles, with moving_margin around them."""
        def laser_angle(x, y):
            a = np.arctan2(y - pose[1], x - pose[0]) - pose[2]
            return np.arctan2(np.sin(a), np.cos(a))

        target = None
        if goal is not None:
            # pure pursuit's steering towards the goal, from the rear axle
            g = self.gap.p
            dx, dy = goal[0] - pose[0], goal[1] - pose[1]
            c, s = math.cos(pose[2]), math.sin(pose[2])
            gx, gy = c * dx + s * dy + g['rear_offset'], -s * dx + c * dy
            target = math.atan(2.0 * g['wheelbase'] * gy / max(gx * gx + gy * gy, 1e-6))
        allowed = pinch = None
        if len(blocking):
            angles = laser_angle(blocking[:, 0], blocking[:, 1])
            ahead = np.abs(angles) < math.pi / 2
            if ahead.any():
                front, pts = angles[ahead], blocking[ahead]
                # the obstacle's edge on the passing side (the raceline's left is the car's
                # left while it drives along the line): pass it at its own distance
                edge = int(np.argmax(front)) if self.side > 0 else int(np.argmin(front))
                allowed = ((front[edge], math.pi) if self.side > 0
                           else (-math.pi, front[edge]))
                pinch = float(np.hypot(pts[edge, 0] - pose[0], pts[edge, 1] - pose[1]))
        extra, margin = None, None
        if len(dynamic):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            d = dynamic - np.asarray(pose[:2])
            extra = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])
            margin = self.p['moving_margin']
        steer, command = self.gap.step(ranges, angle_min, inc, speed, target, allowed, pinch,
                                       extra, margin)
        return steer, min(command, self.p['avoid_max_speed'])


def split_params(params):
    """(Driver params, GapFollow params) out of the controller's parameters: the avoid keys by
    their own names, the gap follow keys with a gap_ prefix."""
    from fwc_controller import gap_follow
    avoid = {k: v for k, v in params.items() if k in DEFAULTS}
    gap = {k[4:]: v for k, v in params.items()
           if k.startswith('gap_') and k[4:] in gap_follow.DEFAULTS}
    return avoid, gap
