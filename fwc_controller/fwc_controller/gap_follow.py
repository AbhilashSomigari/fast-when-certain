"""
Follow the gap (from the course's gap-follow lab), kept free of ROS so it can be tested.

Run on every lidar scan while the controller is avoiding something.

Guided (the controller passes a steering target: pure pursuit's steering towards its goal
point on the raceline) - gap selection over the arcs the car can drive:
  1. arcs        - for each steering angle, the arc the car drives over arc_length m
                   (bicycle model about the rear axle; the laser is rear_offset ahead of it).
  2. clearance   - along each arc, the distance from the car's body (points from just behind
                   the rear axle to just ahead of the laser) to the nearest scan point.
  3. choose      - the steering closest to the target among arcs that keep half the car's
                   width + width_margin / 2 clear, else half the car's width + path_margin,
                   optionally only arcs that pass the obstacle on a given side; if none, the
                   arc that stays clear the longest.
  4. speed       - capped by how far the chosen arc stays clear (time headway).
Choosing among arcs instead of straight headings keeps to paths the car can actually drive
(it can't turn tighter than wheelbase / tan(max_steering)).

Unguided (no target) - the lab's algorithm:
  1. preprocess        - clean NaN/inf, clip far returns, moving-average smoothing, keep only
                         the forward +-fov_deg window.
  2. disparity extender - grow obstacle edges by half the car width so gaps the car cannot
                         fit through disappear.
  3. safety bubble     - beams around the closest point (radius grows with speed) set to 0.
  4. max gap           - longest run of free beams; best point = centre of its deepest part.
  5. corner guard      - don't steer into a wall that is already close beside the car.
  6. speed capped by the free path straight ahead (time headway).

Both: steering smoothed and rate-limited; speed blended on the steering angle, down to
min_speed while the way is clear (0 lets the car stop behind something it can't pass).
"""

import math

import numpy as np

DEFAULTS = {
    'fov_deg': 70.0,
    'max_range': 6.0,
    'smoothing_window': 5,
    'car_width': 0.31,
    'width_margin': 0.25,
    'path_margin': 0.05,
    'disparity_threshold': 0.3,
    'bubble_radius': 0.2,
    'bubble_speed_gain': 0.03,
    'gap_threshold': 2.5,
    'gap_threshold_ratio': 0.7,
    'best_point_window': 40,
    'arc_length': 0.8,
    'wheelbase': 0.3302,
    'rear_offset': 0.27,
    'max_steering': 0.4189,
    'high_speed': 2.0,
    'low_speed': 1.0,
    'min_speed': 0.0,
    'corner_clearance': 0.55,
    'time_headway': 0.3,
    'stop_distance': 0.2,
    'steering_alpha': 0.6,
    'max_steering_step': 0.12,
}


class GapFollow:
    """Steering and speed from one scan. Parameters: DEFAULTS."""

    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError('unknown gap follow parameters: ' + ', '.join(sorted(unknown)))
        self.p = dict(DEFAULTS, **params)
        self.steering = 0.0
        self.lo = self.hi = 0
        p = self.p
        self.steers = np.linspace(-p['max_steering'], p['max_steering'], 31)
        self.s = np.arange(0.1, p['arc_length'] + 1e-9, 0.1)
        # body points, m ahead of the rear axle: rear overhang .. front of the car
        self.body = np.array([-0.08, 0.0, p['rear_offset'] / 2, p['rear_offset'],
                              p['rear_offset'] + 0.1])
        self.arcs = self.arc_points()

    def reset(self, steering=0.0):
        """Start from the steering angle the car has now (e.g. pure pursuit's last one)."""
        self.steering = float(steering)

    # --- guided: arcs ------------------------------------------------------------------

    def arc_points(self):
        """Body points along each arc in the laser frame: shape (steers, s, body, 2), and the
        laser's own position along each arc, going on straight from its end to 4 m."""
        p = self.p
        k = np.tan(self.steers)[:, None] / p['wheelbase']          # curvature
        s = self.s[None, :]
        small = np.abs(k) < 1e-6
        ks = np.where(small, 1.0, k)
        x = np.where(small, s, np.sin(k * s) / ks)
        y = np.where(small, k * s * s / 2.0, (1.0 - np.cos(k * s)) / ks)
        th = k * s
        b = self.body[None, None, :]
        bx = x[..., None] + b * np.cos(th)[..., None] - p['rear_offset']
        by = y[..., None] + b * np.sin(th)[..., None]
        laser = np.stack([x + p['rear_offset'] * np.cos(th) - p['rear_offset'],
                          y + p['rear_offset'] * np.sin(th)], axis=-1)
        # for the side check, each arc goes on straight from its end, to 4 m
        ext = np.arange(0.1, 4.0, 0.1)
        end, heading = laser[:, -1, :], th[:, -1]
        straight = end[:, None, :] + ext[None, :, None] * np.stack(
            [np.cos(heading), np.sin(heading)], axis=-1)[:, None, :]
        return np.stack([bx, by], axis=-1), np.concatenate([laser, straight], axis=1)

    def arc_clearance(self, ranges, angle_min, inc, extra=None):
        """(smallest clearance per arc, clearance per arc and step) from the scan points and
        `extra` points (laser frame, e.g. where the other car will be)."""
        r = np.asarray(ranges, dtype=np.float64)
        a = angle_min + np.arange(r.size) * inc
        ok = np.isfinite(r) & (r > 0.02) & (r < self.p['arc_length'] + 1.0)
        pts = np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])[::2]
        if extra is not None and len(extra):
            near = np.hypot(extra[:, 0], extra[:, 1]) < self.p['arc_length'] + 1.0
            pts = np.vstack([pts, extra[near]])
        body, _ = self.arcs
        if len(pts) == 0:
            far = np.full(body.shape[:2], np.inf)
            return far.min(axis=1), far
        d = np.full(body.shape[:3], np.inf)
        for chunk in np.array_split(pts, max(1, len(pts) // 200)):
            diff = body[..., None, :] - chunk[None, None, None, :, :]
            d = np.minimum(d, np.sqrt((diff ** 2).sum(axis=-1)).min(axis=-1))
        per_step = d.min(axis=2)
        return per_step.min(axis=1), per_step

    def clear_along(self, ranges, angle_min, inc, steering):
        """How far (m, up to arc_length) the car can drive at this steering angle before its
        body comes within 2 cm of a scan point; inf if the whole arc is clear."""
        i = int(np.argmin(np.abs(self.steers - steering)))
        r = np.asarray(ranges, dtype=np.float64)
        a = angle_min + np.arange(r.size) * inc
        ok = np.isfinite(r) & (r > 0.02) & (r < self.p['arc_length'] + 1.0)
        pts = np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])
        if len(pts) == 0:
            return math.inf
        body = self.arcs[0][i]                                  # (s, body, 2)
        d = np.sqrt(((body[:, :, None, :] - pts[None, None]) ** 2).sum(axis=-1)).min(axis=(1, 2))
        hit = np.flatnonzero(d < self.p['car_width'] / 2.0 + 0.02)
        return float(self.s[hit[0]] - 0.1) if len(hit) else math.inf

    def choose_arc(self, ranges, angle_min, inc, target, allowed=None, pinch=None, extra=None,
                   margin=None):
        """(steering, clear length): the arc closest to `target` steering with room (see the
        module docstring). allowed = (low, high) bearings from the laser, pinch = distance:
        only arcs whose laser position at that distance from the laser (the arc going on
        straight past its end) lies at a bearing in the window, with half the car's width +
        path_margin + 5 cm to spare."""
        p = self.p
        room, per_step = self.arc_clearance(ranges, angle_min, inc, extra)
        margin = p['path_margin'] if margin is None else margin
        fit = p['car_width'] / 2.0 + margin
        # how far each arc can be driven: until the body comes within 2 cm of a point
        blocked = per_step < p['car_width'] / 2.0 + 0.02
        clear = np.where(blocked.any(axis=1), self.s[blocked.argmax(axis=1)] - 0.1,
                         p['arc_length'])
        side = np.ones(len(self.steers), dtype=bool)
        if allowed is not None and pinch is not None:
            _, laser = self.arcs
            j = np.argmin(np.abs(np.hypot(laser[..., 0], laser[..., 1]) - pinch), axis=1)
            pt = laser[np.arange(len(self.steers)), j]
            bearing = np.arctan2(pt[:, 1], pt[:, 0])
            # the car's centre line has to clear the edge by half its width + margins
            spare = math.atan2(fit + 0.05, pinch)
            side = (bearing >= allowed[0] + spare) & (bearing <= allowed[1] - spare)
            if not side.any():
                side[:] = True
        for need in (max((p['car_width'] + p['width_margin']) / 2.0, fit), fit):
            ok = side & (room >= need)
            if ok.any():
                i = np.flatnonzero(ok)[np.argmin(np.abs(self.steers[ok] - target))]
                return float(self.steers[i]), float(clear[i])
        i = int(np.argmax(np.where(side, clear, -1.0)))
        return float(self.steers[i]), float(clear[i])

    # --- unguided: the lab's gap follow -------------------------------------------------

    def preprocess(self, ranges, angle_min, inc):
        p = self.p
        r = np.array(ranges, dtype=np.float64)
        r[~np.isfinite(r)] = p['max_range']
        r = np.clip(r, 0.0, p['max_range'])
        w = int(p['smoothing_window'])
        if w > 1:
            pad = w // 2
            r = np.convolve(np.pad(r, (pad, w - 1 - pad), mode='edge'), np.ones(w) / w,
                            mode='valid')
        n = r.size
        fov = math.radians(p['fov_deg'])
        self.lo = int(np.clip(math.ceil((-fov - angle_min) / inc), 0, n - 1))
        self.hi = int(np.clip(math.floor((fov - angle_min) / inc), 0, n - 1))
        r[:self.lo] = 0.0
        r[self.hi + 1:] = 0.0
        return r

    def extend_disparities(self, r, inc):
        out = r.copy()
        half = (self.p['car_width'] + self.p['width_margin']) / 2.0
        window = r[self.lo:self.hi + 1]
        for e in np.where(np.abs(np.diff(window)) > self.p['disparity_threshold'])[0]:
            i = self.lo + int(e)
            near = min(r[i], r[i + 1])
            if near <= 0.0:
                continue
            k = int(math.ceil(math.atan2(half, near) / inc))
            if r[i] < r[i + 1]:
                a, b = i + 1, min(self.hi + 1, i + 1 + k)
            else:
                a, b = max(self.lo, i - k + 1), i + 1
            out[a:b] = np.minimum(out[a:b], near)
        return out

    def max_gap(self, r):
        window = r[self.lo:self.hi + 1]
        deepest = float(window.max()) if window.size else 0.0
        if deepest <= 0.0:
            mid = (self.lo + self.hi) // 2
            return mid, mid
        free = window > min(self.p['gap_threshold'], self.p['gap_threshold_ratio'] * deepest)
        best, best_len, best_depth = (self.lo, self.lo), 0, -1.0
        start = None
        for offset, is_free in enumerate(np.append(free, False)):
            i = self.lo + offset
            if is_free:
                start = i if start is None else start
                continue
            if start is not None:
                length, depth = i - start, float(r[start:i].sum())
                if length > best_len or (length == best_len and depth > best_depth):
                    best, best_len, best_depth = (start, i - 1), length, depth
                start = None
        return best

    def best_angle(self, start, end, r, angle_min, inc):
        gap = r[start:end + 1]
        w = max(1, min(int(self.p['best_point_window']), gap.size))
        smoothed = np.convolve(gap, np.ones(w) / w, mode='same')
        ties = np.where(smoothed >= smoothed.max() - 1e-3)[0]
        blocks = np.split(ties, np.where(np.diff(ties) > 1)[0] + 1)
        block = max(blocks, key=len)
        return angle_min + (start + int(block[len(block) // 2])) * inc

    def free_path_ahead(self, r, angle_min, inc, heading=0.0):
        """Length of the free corridor (car width + path_margin each side) from the laser in
        direction `heading`."""
        half = self.p['car_width'] / 2.0 + self.p['path_margin']
        idx = np.arange(self.lo, self.hi + 1)
        a = angle_min + idx * inc - heading
        x, y = r[idx] * np.cos(a), r[idx] * np.sin(a)
        blocking = (r[idx] > 0.0) & (x > 0.0) & (np.abs(y) < half)
        return float(x[blocking].min()) if blocking.any() else self.p['max_range']

    def side_clearance(self, ranges, angle_min, inc):
        r = np.asarray(ranges, dtype=np.float64)
        a = angle_min + np.arange(r.size) * inc
        valid = np.isfinite(r) & (r > 0.0)
        lo, hi = math.radians(60.0), math.radians(120.0)
        left, right = valid & (a >= lo) & (a <= hi), valid & (a <= -lo) & (a >= -hi)
        far = self.p['max_range']
        return (float(r[left].min()) if left.any() else far,
                float(r[right].min()) if right.any() else far)

    def corner_guard(self, steering, left, right):
        half = self.p['car_width'] / 2.0
        side = left if steering > 0.0 else right
        if side >= self.p['corner_clearance']:
            return steering
        scale = (side - half) / (self.p['corner_clearance'] - half)
        return steering * float(np.clip(scale, 0.0, 1.0))

    def lab_steering(self, ranges, angle_min, inc, speed):
        """(steering target, free path ahead) from the lab's algorithm."""
        p = self.p
        clean = self.preprocess(ranges, angle_min, inc)
        r = self.extend_disparities(clean, inc)
        window = r[self.lo:self.hi + 1]
        closest_i = self.lo + int(np.argmin(window))
        closest = float(r[closest_i])
        radius = p['bubble_radius'] + p['bubble_speed_gain'] * abs(speed)
        half_angle = math.pi / 4 if closest <= radius else math.asin(radius / closest)
        k = int(math.ceil(half_angle / inc))
        free = r.copy()
        free[max(self.lo, closest_i - k):min(self.hi, closest_i + k) + 1] = 0.0
        start, end = self.max_gap(free)
        aim = float(np.clip(self.best_angle(start, end, free, angle_min, inc),
                            -p['max_steering'], p['max_steering']))
        aim = self.corner_guard(aim, *self.side_clearance(ranges, angle_min, inc))
        return aim, self.free_path_ahead(clean, angle_min, inc)

    # --- both -------------------------------------------------------------------------

    def step(self, ranges, angle_min, inc, speed, target=None, allowed=None, pinch=None,
             extra=None, margin=None):
        """Return (steering angle, speed command) for one scan. speed = current speed,
        target = preferred steering angle (rad), e.g. pure pursuit's towards its goal point;
        allowed = (low, high) bearings and pinch = distance (m): only arcs that are at a
        bearing in that window at that distance, e.g. one side of an obstacle; extra = more
        obstacle points (laser frame), margin = room kept beside the car instead of
        path_margin."""
        p = self.p
        if target is not None:
            aim, free = self.choose_arc(ranges, angle_min, inc, target, allowed, pinch, extra,
                                        margin)
        else:
            aim, free = self.lab_steering(ranges, angle_min, inc, speed)
        smoothed = p['steering_alpha'] * aim + (1.0 - p['steering_alpha']) * self.steering
        self.steering += float(np.clip(smoothed - self.steering,
                                       -p['max_steering_step'], p['max_steering_step']))

        turn = min(abs(self.steering) / p['max_steering'], 1.0)
        command = p['high_speed'] - (p['high_speed'] - p['low_speed']) * turn
        ahead = free - p['stop_distance']
        if p['time_headway'] > 0.0:
            floor = p['min_speed'] if ahead > 0.0 else 0.0
            command = min(command, max(floor, ahead / p['time_headway']))
        return self.steering, max(command, 0.0)
