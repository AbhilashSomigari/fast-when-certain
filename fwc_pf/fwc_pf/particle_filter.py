# MIT License

# Copyright (c) 2020 Hongrui Zheng, Corey Walsh, Chinmay Samak, Tanmay Samak

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Particle filter node: ROS around mcl.ParticleFilter, with range_libc for ray casting.

Waits for a 2D Pose Estimate (/initialpose) before it does anything; Publish Point
(/clicked_point) spreads the particles over the whole map instead. Then one update per
odometry message, using the latest scan.

Publishes
  /pf/pose/odom       pose of the laser in the map (twist = odometry speed)
  /pf/stats           [inlier, spread, ess], see mcl.py
  /pf/confidence      0..1 from the inlier ratio (conf_low, conf_high, conf_alpha)
  TF map -> pf_laser  and the measured scan in that frame (/pf/viz/scan), so RViz can show
                      the scan against the map without touching the car's own TF tree
  /pf/viz/...         particles, pose, expected scan, for RViz
"""

import time

import numpy as np
import range_libc
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import (PointStamped, Pose, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped, Quaternion, TransformStamped)
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray
from tf2_ros import TransformBroadcaster

from fwc_pf import mcl

# ROS parameter -> default; the mcl ones are passed on to mcl.ParticleFilter
PARAMS = {
    'scan_topic': '/scan',
    'odometry_topic': '/odom',
    'angle_step': 18,
    'range_method': 'cddt',
    'theta_discretization': 112,
    'max_viz_particles': 60,
    'viz': True,
    'conf_low': 0.7,
    'conf_high': 0.95,
    'conf_alpha': 0.3,
}
PARAMS.update(mcl.DEFAULTS)
MAP_FRAME = 'map'
LASER_FRAME = 'pf_laser'


def yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw / 2.0)), w=float(np.cos(yaw / 2.0)))


def quaternion_to_yaw(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class ParticleFilterNode(Node):

    def __init__(self):
        super().__init__('particle_filter')
        self.p = {name: self.declare_parameter(name, default).value
                  for name, default in PARAMS.items()}

        grid = self.load_map()
        pf_params = {k: self.p[k] for k in mcl.DEFAULTS}
        self.pf = mcl.ParticleFilter(grid, self.raycast, **pf_params)
        self.confidence = mcl.Confidence(self.p['conf_low'], self.p['conf_high'],
                                         self.p['conf_alpha'])

        self.scan = None          # latest LaserScan
        self.angles = None        # downsampled beam angles (rad)
        self.ranges = None        # downsampled measured ranges (m)
        self.odom_anchor = None   # odometry pose (x, y, yaw) at the last update
        self.speed = 0.0
        self.update_times = []

        self.pub_odom = self.create_publisher(Odometry, '/pf/pose/odom', 1)
        self.pub_stats = self.create_publisher(Float32MultiArray, '/pf/stats', 1)
        self.pub_confidence = self.create_publisher(Float32, '/pf/confidence', 1)
        self.pub_pose = self.create_publisher(PoseStamped, '/pf/viz/inferred_pose', 1)
        self.pub_particles = self.create_publisher(PoseArray, '/pf/viz/particles', 1)
        self.pub_scan = self.create_publisher(LaserScan, '/pf/viz/scan', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, '/pf/viz/fake_scan', 1)
        self.tf = TransformBroadcaster(self)

        self.create_subscription(LaserScan, self.p['scan_topic'], self.scan_callback, 1)
        self.create_subscription(Odometry, self.p['odometry_topic'], self.odom_callback, 1)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.pose_callback, 1)
        self.create_subscription(PointStamped, '/clicked_point', self.point_callback, 1)
        self.get_logger().info('Ready (%d particles, %s ray casting). Waiting for a 2D Pose '
                               'Estimate.' % (self.pf.n, self.p['range_method']))

    def load_map(self):
        client = self.create_client(GetMap, '/map_server/map')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the map server...')
        future = client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        msg = future.result().map
        info = msg.info
        max_px = int(self.p['max_range'] / info.resolution)
        omap = range_libc.PyOMap(msg)
        method = self.p['range_method']
        if method in ('cddt', 'pcddt'):
            self.range_method = range_libc.PyCDDTCast(omap, max_px, self.p['theta_discretization'])
            if method == 'pcddt':
                self.range_method.prune()
        elif method == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(omap, max_px)
        elif method == 'rm':
            self.range_method = range_libc.PyRayMarching(omap, max_px)
        elif method == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(omap, max_px)
        else:
            raise ValueError('range_method must be cddt, pcddt, rmgpu, rm or bl')
        free = np.array(msg.data).reshape(info.height, info.width) == 0
        origin = (info.origin.position.x, info.origin.position.y,
                  quaternion_to_yaw(info.origin.orientation))
        self.get_logger().info('Map %dx%d cells at %.3f m'
                               % (info.width, info.height, info.resolution))
        return mcl.GridMap(free, info.resolution, origin)

    def raycast(self, poses, angles):
        queries = np.ascontiguousarray(poses, dtype=np.float32)
        angles = np.ascontiguousarray(angles, dtype=np.float32)
        out = np.empty(len(queries) * len(angles), dtype=np.float32)
        self.range_method.calc_range_repeat_angles(queries, angles, out)
        return out.reshape(len(queries), len(angles))

    # callbacks

    def scan_callback(self, msg):
        step = int(self.p['angle_step'])
        if self.angles is None:
            angles = msg.angle_min + msg.angle_increment * np.arange(len(msg.ranges))
            self.angles = angles[::step].astype(np.float32)
        self.ranges = np.array(msg.ranges[::step], dtype=float)
        self.scan = msg

    def pose_callback(self, msg):
        p = msg.pose.pose
        self.get_logger().info('2D Pose Estimate at (%.2f, %.2f)' % (p.position.x, p.position.y))
        self.pf.init_pose(p.position.x, p.position.y, quaternion_to_yaw(p.orientation))
        # the sim moves the car on /initialpose: don't count that jump as motion
        self.odom_anchor = None
        self.confidence.reset()

    def point_callback(self, msg):
        self.get_logger().info('Publish Point: particles spread over the whole map')
        self.pf.init_global()
        self.odom_anchor = None
        self.confidence.reset()

    def odom_callback(self, msg):
        p = msg.pose.pose
        pose = np.array([p.position.x, p.position.y, quaternion_to_yaw(p.orientation)])
        self.speed = msg.twist.twist.linear.x
        if self.odom_anchor is None or not self.pf.initialized or self.ranges is None:
            self.odom_anchor = pose
            return
        action = mcl.odom_delta(self.odom_anchor, pose)
        self.odom_anchor = pose
        self.update(action, msg.header.stamp)

    # one filter step and its outputs

    def update(self, action, stamp):
        t0 = time.time()
        pose = self.pf.update(action, self.speed, self.ranges, self.angles)
        confidence = self.confidence.update(self.pf.stats[0])
        self.log_rate(time.time() - t0)

        self.publish_tf(pose, stamp)
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = MAP_FRAME
        odom.child_frame_id = LASER_FRAME
        odom.pose.pose = self.to_pose(pose)
        cov = np.cov(self.pf.particles.T, aweights=self.pf.weights, ddof=0)
        for i, a in enumerate((0, 1, 5)):          # x, y, yaw in the 6x6 covariance
            for j, b in enumerate((0, 1, 5)):
                odom.pose.covariance[6 * a + b] = float(cov[i, j])
        odom.twist.twist.linear.x = float(self.speed)
        self.pub_odom.publish(odom)
        self.pub_stats.publish(Float32MultiArray(data=[float(s) for s in self.pf.stats]))
        self.pub_confidence.publish(Float32(data=float(confidence)))
        if self.p['viz']:
            self.visualize(pose, stamp)

    def log_rate(self, compute_time):
        now = time.time()
        self.update_times.append((now, compute_time))
        if now - self.update_times[0][0] >= 2.0:
            n = len(self.update_times)
            rate = (n - 1) / (now - self.update_times[0][0])
            possible = 1.0 / np.mean([c for _, c in self.update_times])
            self.get_logger().info('iters per sec: %d  possible: %d' % (rate, possible))
            self.update_times = []

    def to_pose(self, pose):
        msg = Pose()
        msg.position.x = float(pose[0])
        msg.position.y = float(pose[1])
        msg.orientation = yaw_to_quaternion(pose[2])
        return msg

    def publish_tf(self, pose, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = MAP_FRAME
        t.child_frame_id = LASER_FRAME
        t.transform.translation.x = float(pose[0])
        t.transform.translation.y = float(pose[1])
        t.transform.rotation = yaw_to_quaternion(pose[2])
        self.tf.sendTransform(t)

    def visualize(self, pose, stamp):
        if self.pub_pose.get_subscription_count() > 0:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = MAP_FRAME
            ps.pose = self.to_pose(pose)
            self.pub_pose.publish(ps)
        if self.pub_particles.get_subscription_count() > 0:
            n = min(self.pf.n, int(self.p['max_viz_particles']))
            idx = self.pf.rng.choice(self.pf.n, n, p=self.pf.weights)
            pa = PoseArray()
            pa.header.stamp = stamp
            pa.header.frame_id = MAP_FRAME
            pa.poses = [self.to_pose(q) for q in self.pf.particles[idx]]
            self.pub_particles.publish(pa)
        if self.pub_scan.get_subscription_count() > 0:
            scan = self.scan
            scan.header.stamp = stamp
            scan.header.frame_id = LASER_FRAME
            self.pub_scan.publish(scan)
        if self.pub_fake_scan.get_subscription_count() > 0:
            fake = LaserScan()
            fake.header.stamp = stamp
            fake.header.frame_id = LASER_FRAME
            fake.angle_min = float(self.angles[0])
            fake.angle_max = float(self.angles[-1])
            fake.angle_increment = float(self.angles[1] - self.angles[0])
            fake.range_min = 0.0
            fake.range_max = float(self.p['max_range'])
            fake.ranges = [float(r) for r in self.pf.expected_ranges]
            self.pub_fake_scan.publish(fake)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
