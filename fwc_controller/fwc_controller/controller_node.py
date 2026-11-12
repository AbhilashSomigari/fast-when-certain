#!/usr/bin/env python3
"""
Pure pursuit on the particle filter pose, with optional confidence-aware speed.

On start-up the waypoint CSV is cleaned, cut to one lap, resampled, smoothed
and given a speed profile (pure_pursuit.prepare_path, speed_profile). Every
pose on /pf/pose/odom goes through pure_pursuit.PurePursuit.step and out as
an AckermannDriveStamped on /drive. No pose, no drive command: the filter
publishes nothing until it has a 2D Pose Estimate.

With adaptive_speed on, the latest /pf/confidence is used; if it is older
than conf_timeout it counts as 0.
"""

import os

from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker

from fwc_controller import pure_pursuit as pp

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
}


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

        self.path_marker = self.make_marker(0, 0.1, (0.0, 0.0, 1.0))
        self.path_marker.points = [
            Point(x=float(x), y=float(y), z=0.0) for x, y in path]
        self.create_timer(1.0, self.publish_path)

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
        angle, speed, goal = self.pursuit.step(
            pose.position.x, pose.position.y, yaw,
            msg.twist.twist.linear.x, self.fresh_confidence(now), dt)

        drive = AckermannDriveStamped()
        drive.header.stamp = now.to_msg()
        drive.drive.steering_angle = float(angle)
        drive.drive.speed = float(speed)
        self.pub_drive.publish(drive)

        marker = self.make_marker(1, 0.2, (1.0, 0.0, 0.0))
        marker.header.stamp = now.to_msg()
        marker.points = [Point(x=float(goal[0]), y=float(goal[1]), z=0.0)]
        self.pub_dynamic_viz.publish(marker)

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
