#!/usr/bin/env python3
"""
Pure pursuit waypoint follower with optional confidence-aware speed.

Ported from the Lab 4 pure pursuit node. On start-up the waypoint CSV is
cleaned, cut to one lap, resampled, smoothed and given a speed profile (see
pure_pursuit.py). On every pose message:
  1. nearest waypoint (windowed search ahead of the previous one),
  2. target speed = speed_scale * profile speed at that waypoint,
  3. with `adaptive_speed` on: confidence c from /pf/confidence (0 if it is
     stale), allowed to fall at once but to rise by at most conf_rise_rate
     per second; speed blended from conf_min_speed (c = 0) to the target
     (c = 1), and the lookahead lengthened by lookahead_conf_gain * (1 - c),
  4. goal point at the lookahead distance L = lookahead_distance
     + lookahead_speed_gain * speed (+ the confidence term),
  5. curvature gamma = 2 y / L^2 in the vehicle frame,
     steering = atan(wheelbase * gamma), clipped to the steering limit,
  6. AckermannDriveStamped on /drive, waypoints and goal point as RViz
     markers.

Pose comes from the particle filter (/pf/pose/odom).
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


class Controller(Node):
    """Pure pursuit on the particle filter pose."""

    def __init__(self):
        super().__init__('controller')

        # Topics
        odom_topic = self.declare_parameter(
            'odom_topic', '/pf/pose/odom').value
        drive_topic = self.declare_parameter('drive_topic', '/drive').value
        confidence_topic = self.declare_parameter(
            'confidence_topic', '/pf/confidence').value
        self.frame_id = self.declare_parameter('frame_id', 'map').value

        # Waypoints (a file name in the package's waypoints/ folder, or a
        # full path)
        waypoints = self.declare_parameter('waypoints', 'track.csv').value
        spacing = self.declare_parameter('waypoint_spacing', 0.05).value
        smoothing_window = self.declare_parameter(
            'smoothing_window', 9).value
        self.search_window = self.declare_parameter(
            'search_window', 100).value

        # Lookahead and steering
        self.lookahead_distance = self.declare_parameter(
            'lookahead_distance', 0.9).value
        self.lookahead_speed_gain = self.declare_parameter(
            'lookahead_speed_gain', 0.05).value
        self.wheelbase = self.declare_parameter('wheelbase', 0.3302).value
        self.max_steering = self.declare_parameter(
            'max_steering', 0.4189).value

        # Speed profile
        max_speed = self.declare_parameter('max_speed', 5.0).value
        min_speed = self.declare_parameter('min_speed', 1.5).value
        max_lateral_accel = self.declare_parameter(
            'max_lateral_accel', 5.0).value
        max_decel = self.declare_parameter('max_decel', 5.0).value
        self.speed_scale = self.declare_parameter('speed_scale', 1.0).value

        # Confidence-aware speed and lookahead
        self.adaptive_speed = self.declare_parameter(
            'adaptive_speed', False).value
        self.conf_min_speed = self.declare_parameter(
            'conf_min_speed', 0.8).value
        self.conf_rise_rate = self.declare_parameter(
            'conf_rise_rate', 0.5).value
        self.conf_timeout = self.declare_parameter('conf_timeout', 0.5).value
        self.lookahead_conf_gain = self.declare_parameter(
            'lookahead_conf_gain', 0.3).value

        if not os.path.isabs(waypoints):
            waypoints = os.path.join(
                get_package_share_directory('fwc_controller'),
                'waypoints', waypoints)
        raw = pp.load_waypoints(waypoints)
        self.path = pp.prepare_path(raw, spacing, smoothing_window)
        self.speeds = pp.speed_profile(
            self.path, max_speed, min_speed, max_lateral_accel, max_decel)
        self.get_logger().info(
            'Loaded %d waypoints from %s (%d after resampling), '
            'speed %.2f-%.2f m/s x %.2f, adaptive_speed %s'
            % (len(raw), waypoints, len(self.path), self.speeds.min(),
               self.speeds.max(), self.speed_scale, self.adaptive_speed))

        self.current_idx = None
        self.current_speed = 0.0
        # Latest confidence and when it arrived; the confidence actually
        # used starts at 0 so the car pulls away slowly and ramps up.
        self.confidence = 0.0
        self.confidence_time = None
        self.used_confidence = 0.0
        self.last_pose_time = None

        # Subs, Pubs
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)
        self.pub_env_viz = self.create_publisher(Marker, '/env_viz', 10)
        self.pub_dynamic_viz = self.create_publisher(
            Marker, '/dynamic_viz', 10)
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.pose_callback, 10)
        if self.adaptive_speed:
            self.sub_confidence = self.create_subscription(
                Float32, confidence_topic, self.confidence_callback, 10)

        self.path_marker = self.make_marker(0, 0.1, (0.0, 0.0, 1.0))
        self.path_marker.points = [
            Point(x=float(x), y=float(y), z=0.0) for x, y in self.path]
        self.create_timer(1.0, self.publish_path)

    def confidence_callback(self, msg):
        self.confidence = min(max(float(msg.data), 0.0), 1.0)
        self.confidence_time = self.get_clock().now()

    def update_confidence(self, now):
        """Confidence to use now: 0 if stale, rises at a limited rate."""
        fresh = (self.confidence_time is not None
                 and (now - self.confidence_time).nanoseconds * 1e-9
                 < self.conf_timeout)
        target = self.confidence if fresh else 0.0
        dt = 0.0
        if self.last_pose_time is not None:
            dt = (now - self.last_pose_time).nanoseconds * 1e-9
        self.used_confidence = pp.limit_rise(
            self.used_confidence, target, self.conf_rise_rate * dt)

    def pose_callback(self, odometry_info):
        now = self.get_clock().now()
        pose = odometry_info.pose.pose
        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
        yaw = pp.yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_speed = odometry_info.twist.twist.linear.x

        # Current waypoint: closest one, searched ahead of the last one.
        self.current_idx = pp.nearest_index(
            self.path, x, y, self.current_idx, self.search_window)

        # Speed from the profile, slowed down when the filter is unsure.
        speed = self.speed_scale * float(self.speeds[self.current_idx])
        lookahead = (self.lookahead_distance
                     + self.lookahead_speed_gain * abs(self.current_speed))
        if self.adaptive_speed:
            self.update_confidence(now)
            c = self.used_confidence
            speed = pp.confident_speed(speed, c, self.conf_min_speed)
            lookahead += self.lookahead_conf_gain * (1.0 - c)
        self.last_pose_time = now

        # Goal point on the path at the lookahead distance.
        goal, _ = pp.lookahead_point(
            self.path, x, y, self.current_idx, lookahead)

        # Goal point in the vehicle frame, then curvature -> steering.
        local_x, local_y = pp.to_vehicle_frame(goal, x, y, yaw)
        angle = pp.steering_angle(
            local_x, local_y, self.wheelbase, self.max_steering)

        self.publish_drive(angle, speed)
        self.publish_goal(goal)

    def publish_drive(self, angle, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(angle)
        msg.drive.speed = float(speed)
        self.pub_drive.publish(msg)

    def make_marker(self, marker_id, scale, rgb):
        marker = Marker()
        marker.header.frame_id = self.frame_id
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

    def publish_goal(self, goal):
        marker = self.make_marker(1, 0.2, (1.0, 0.0, 0.0))
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.points.append(
            Point(x=float(goal[0]), y=float(goal[1]), z=0.0))
        self.pub_dynamic_viz.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
