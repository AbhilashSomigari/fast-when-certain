#!/usr/bin/env python3
"""
Pure pursuit on the particle filter pose, with optional confidence-aware speed,
and follow the gap around obstacles that aren't in the map.

On start-up the waypoint CSV is cleaned, cut to one lap, resampled, smoothed
and given a speed profile (pure_pursuit.prepare_path, speed_profile). Every
pose on /pf/pose/odom goes through pure_pursuit.PurePursuit.step. No pose, no
drive command: the filter publishes nothing until it has a 2D Pose Estimate.

With adaptive_speed on, the latest /pf/confidence is used; if it is older
than conf_timeout it counts as 0.

With avoid on (avoid.Driver), every /scan is checked for obstacles on the
raceline ahead (the map comes from /map_server/map, like the filter's):
  mode TRACK  - pure pursuit's command goes out on /drive (on every pose)
  mode AVOID  - gap follow's command goes out on /drive (on every scan)
  mode FOLLOW - pure pursuit steers, at most the speed that keeps the gap to
                the other car (on every pose)
In every mode the speed is 0 while the time-to-collision brake is on.
Published: /controller/mode (Int8, 0 TRACK / 1 AVOID / 2 FOLLOW),
/controller/offset (Float32, m from the raceline), on /dynamic_viz: obstacle
points (id 2, orange) and the other car's predicted points (id 3, magenta).
"""

import os

from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int8
from visualization_msgs.msg import Marker

from fwc_controller import avoid
from fwc_controller import gap_follow
from fwc_controller import pure_pursuit as pp

MODES = {avoid.TRACK: 'TRACK', avoid.AVOID: 'AVOID', avoid.FOLLOW: 'FOLLOW'}

PARAMS = {
    'odom_topic': '/pf/pose/odom',
    'drive_topic': '/drive',
    'confidence_topic': '/pf/confidence',
    'frame_id': 'map',
    'waypoints': 'track.csv',
    'waypoint_spacing': 0.05,
    'smoothing_window': 9,
    'search_window': 100,
    'lookahead_distance': 0.9,
    'lookahead_speed_gain': 0.05,
    'wheelbase': 0.3302,
    'max_steering': 0.4189,
    'max_speed': 5.0,
    'min_speed': 1.5,
    'max_lateral_accel': 5.0,
    'max_decel': 5.0,
    'speed_scale': 1.0,
    'adaptive_speed': False,
    'conf_min_speed': 0.8,
    'conf_rise_rate': 0.5,
    'conf_timeout': 0.5,
    'lookahead_conf_gain': 0.3,
    'scan_topic': '/scan',
    'avoid': False,
}
PARAMS.update(avoid.DEFAULTS)
PARAMS.update({'gap_' + k: v for k, v in gap_follow.DEFAULTS.items()})


class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        p = {name: self.declare_parameter(name, default).value
             for name, default in PARAMS.items()}
        self.p = p

        waypoints = p['waypoints']
        if not os.path.isabs(waypoints):
            waypoints = os.path.join(
                get_package_share_directory('fwc_controller'),
                'waypoints', waypoints)
        raw = pp.load_waypoints(waypoints)
        path = pp.prepare_path(raw, p['waypoint_spacing'],
                               p['smoothing_window'])
        speeds = pp.speed_profile(path, p['max_speed'], p['min_speed'],
                                  p['max_lateral_accel'], p['max_decel'])
        self.pursuit = pp.PurePursuit(
            path, speeds, p['lookahead_distance'], p['lookahead_speed_gain'],
            p['wheelbase'], p['max_steering'], p['speed_scale'],
            p['search_window'], p['adaptive_speed'], p['conf_min_speed'],
            p['conf_rise_rate'], p['lookahead_conf_gain'])
        self.get_logger().info(
            'Loaded %d waypoints from %s (%d after resampling), '
            'speed %.2f-%.2f m/s x %.2f, adaptive_speed %s'
            % (len(raw), waypoints, len(path), speeds.min(), speeds.max(),
               p['speed_scale'], p['adaptive_speed']))

        self.confidence = None
        self.confidence_time = None
        self.last_pose_time = None
        self.pose = None              # latest laser pose (x, y, yaw) and speed
        self.speed = 0.0
        self.goal = None
        self.steering = 0.0
        self.mode = avoid.TRACK

        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, p['drive_topic'], 10)
        self.pub_env_viz = self.create_publisher(Marker, '/env_viz', 10)
        self.pub_dynamic_viz = self.create_publisher(
            Marker, '/dynamic_viz', 10)
        self.create_subscription(
            Odometry, p['odom_topic'], self.pose_callback, 10)
        if p['adaptive_speed']:
            self.create_subscription(
                Float32, p['confidence_topic'], self.confidence_callback, 10)
        self.driver = None
        if p['avoid']:
            avoid_params, gap_params = avoid.split_params(p)
            self.driver = avoid.Driver(path, self.load_wall_mask(),
                                       gap_follow.GapFollow(**gap_params),
                                       **avoid_params)
            self.pub_mode = self.create_publisher(Int8, '/controller/mode', 10)
            self.pub_offset = self.create_publisher(
                Float32, '/controller/offset', 10)
            self.create_subscription(
                LaserScan, p['scan_topic'], self.scan_callback, 10)
            self.get_logger().info('Obstacle avoidance on (follow the gap)')

        self.path_marker = self.make_marker(0, 0.1, (0.0, 0.0, 1.0))
        self.path_marker.points = [
            Point(x=float(x), y=float(y), z=0.0) for x, y in path]
        self.create_timer(1.0, self.publish_path)

    def load_wall_mask(self):
        """The map's walls, from the map server (as the filter loads it)."""
        client = self.create_client(GetMap, '/map_server/map')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the map server...')
        future = client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        grid = future.result().map
        info = grid.info
        q = info.origin.orientation
        origin = (info.origin.position.x, info.origin.position.y,
                  pp.yaw_from_quaternion(q.x, q.y, q.z, q.w))
        occupied = avoid.occupied_from_grid(grid.data, info.width, info.height)
        return avoid.WallMask(occupied, info.resolution, origin,
                              self.p['wall_margin'])

    def confidence_callback(self, msg):
        self.confidence = float(msg.data)
        self.confidence_time = self.get_clock().now()

    def fresh_confidence(self, now):
        if self.confidence_time is None:
            return None
        age = (now - self.confidence_time).nanoseconds * 1e-9
        return self.confidence if age < self.p['conf_timeout'] else None

    def pose_callback(self, msg):
        now = self.get_clock().now()
        dt = 0.0
        if self.last_pose_time is not None:
            dt = (now - self.last_pose_time).nanoseconds * 1e-9
        self.last_pose_time = now
        pose = msg.pose.pose
        q = pose.orientation
        yaw = pp.yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.speed = msg.twist.twist.linear.x
        self.pose = (pose.position.x, pose.position.y, yaw)
        angle, speed, goal = self.pursuit.step(
            pose.position.x, pose.position.y, yaw,
            self.speed, self.fresh_confidence(now), dt)
        self.goal = goal

        if self.mode == avoid.FOLLOW and self.driver.follow_speed is not None:
            speed = min(speed, self.driver.follow_speed)
        if self.mode != avoid.AVOID:
            self.publish_drive(angle, speed, now)
        if self.driver is not None:
            near = self.pursuit.path[self.pursuit.index]
            self.pub_offset.publish(Float32(data=float(np.hypot(
                near[0] - pose.position.x, near[1] - pose.position.y))))

        marker = self.make_marker(1, 0.2, (1.0, 0.0, 0.0))
        marker.header.stamp = now.to_msg()
        marker.points = [Point(x=float(goal[0]), y=float(goal[1]), z=0.0)]
        self.pub_dynamic_viz.publish(marker)

    def scan_callback(self, msg):
        if self.pose is None:          # not localized yet: nothing drives
            return
        now = self.get_clock().now()
        mode, command, points = self.driver.step(
            now.nanoseconds * 1e-9, self.pose, self.pursuit.index, self.speed,
            msg.ranges, msg.angle_min, msg.angle_increment, self.steering,
            self.goal)
        if mode != self.mode:
            self.get_logger().info('mode %s -> %s' % (MODES[self.mode], MODES[mode]))
        self.mode = mode
        self.pub_mode.publish(Int8(data=int(self.mode)))
        if self.mode == avoid.AVOID:
            self.publish_drive(command[0], command[1], now)
        elif self.driver.braking:
            self.publish_drive(self.steering, 0.0, now)
        marker = self.make_marker(2, 0.05, (1.0, 0.5, 0.0))
        marker.header.stamp = now.to_msg()
        marker.points = [Point(x=float(x), y=float(y), z=0.0)
                         for x, y in points[::4]]
        self.pub_dynamic_viz.publish(marker)
        marker = self.make_marker(3, 0.05, (1.0, 0.0, 1.0))
        marker.header.stamp = now.to_msg()
        marker.points = [Point(x=float(x), y=float(y), z=0.0)
                         for x, y in self.driver.predicted[::4]]
        self.pub_dynamic_viz.publish(marker)

    def publish_drive(self, steering, speed, now):
        if self.driver is not None and self.driver.braking:
            speed = 0.0
        self.steering = float(steering)
        drive = AckermannDriveStamped()
        drive.header.stamp = now.to_msg()
        drive.drive.steering_angle = float(steering)
        drive.drive.speed = float(speed)
        self.pub_drive.publish(drive)

    def make_marker(self, marker_id, scale, rgb):
        marker = Marker()
        marker.header.frame_id = self.p['frame_id']
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = rgb
        return marker

    def publish_path(self):
        self.path_marker.header.stamp = self.get_clock().now().to_msg()
        self.pub_env_viz.publish(self.path_marker)


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
