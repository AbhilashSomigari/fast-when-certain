"""
Stand-ins for rclpy, the message packages, tf2_ros and range_libc, so the two ROS nodes can
run on the Mac. Messages are plain attribute holders; publishing delivers to every
subscription on the topic at once (one synchronous bus); time is set by the test.

Use: fake_ros.install(world) before importing a node module; fake_ros.BUS, fake_ros.CLOCK.
This checks our wiring (topics, parameters, callbacks, message fields), not ROS itself.
"""

import sys
import types

import numpy as np

REPO = None
WORLD = None
OVERRIDES = {}     # node name -> {parameter: value}, like a launch file's parameters=[...]


class Msg:
    FIELDS = {}

    def __init__(self, **kw):
        for name, default in self.FIELDS.items():
            setattr(self, name, default() if callable(default) else default)
        for name, value in kw.items():
            if name not in self.FIELDS:
                raise AttributeError('%s has no field %s' % (type(self).__name__, name))
            setattr(self, name, value)

    def __setattr__(self, name, value):
        if name not in self.FIELDS:
            raise AttributeError('%s has no field %s' % (type(self).__name__, name))
        object.__setattr__(self, name, value)


def msg(name, **fields):
    return type(name, (Msg,), {'FIELDS': fields})


Time = msg('Time', sec=0, nanosec=0)
Header = msg('Header', stamp=Time, frame_id='')
Vector3 = msg('Vector3', x=0.0, y=0.0, z=0.0)
Point = msg('Point', x=0.0, y=0.0, z=0.0)
Quaternion = msg('Quaternion', x=0.0, y=0.0, z=0.0, w=1.0)
Pose = msg('Pose', position=Point, orientation=Quaternion)
PoseStamped = msg('PoseStamped', header=Header, pose=Pose)
PoseArray = msg('PoseArray', header=Header, poses=list)
PoseWithCovariance = msg('PoseWithCovariance', pose=Pose, covariance=lambda: [0.0] * 36)
PoseWithCovarianceStamped = msg('PoseWithCovarianceStamped', header=Header,
                                pose=PoseWithCovariance)
PointStamped = msg('PointStamped', header=Header, point=Point)
Transform = msg('Transform', translation=Vector3, rotation=Quaternion)
TransformStamped = msg('TransformStamped', header=Header, child_frame_id='', transform=Transform)
Twist = msg('Twist', linear=Vector3, angular=Vector3)
TwistWithCovariance = msg('TwistWithCovariance', twist=Twist, covariance=lambda: [0.0] * 36)
Odometry = msg('Odometry', header=Header, child_frame_id='', pose=PoseWithCovariance,
               twist=TwistWithCovariance)
LaserScan = msg('LaserScan', header=Header, angle_min=0.0, angle_max=0.0, angle_increment=0.0,
                time_increment=0.0, scan_time=0.0, range_min=0.0, range_max=0.0,
                ranges=list, intensities=list)
Float32 = msg('Float32', data=0.0)
Float32MultiArray = msg('Float32MultiArray', layout=None, data=list)
MapMetaData = msg('MapMetaData', map_load_time=Time, resolution=0.0, width=0, height=0,
                  origin=Pose)
OccupancyGrid = msg('OccupancyGrid', header=Header, info=MapMetaData, data=list)
AckermannDrive = msg('AckermannDrive', steering_angle=0.0, steering_angle_velocity=0.0,
                     speed=0.0, acceleration=0.0, jerk=0.0)
AckermannDriveStamped = msg('AckermannDriveStamped', header=Header, drive=AckermannDrive)
ColorRGBA = msg('ColorRGBA', r=0.0, g=0.0, b=0.0, a=0.0)
Marker = msg('Marker', header=Header, ns='', id=0, type=0, action=0, pose=Pose, scale=Vector3,
             color=ColorRGBA, points=list)
Marker.POINTS, Marker.ADD = 8, 0


class GetMap:
    Request = msg('GetMapRequest')


class Bus:
    def __init__(self):
        self.subs = {}       # topic -> callbacks
        self.log = {}        # topic -> published messages

    def publish(self, topic, message):
        self.log.setdefault(topic, []).append(message)
        for cb in list(self.subs.get(topic, [])):
            cb(message)


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return FakeTime(self.t)


class FakeTime:
    def __init__(self, t):
        self.nanoseconds = int(round(t * 1e9))

    def __sub__(self, other):
        return types.SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)

    def to_msg(self):
        return Time(sec=self.nanoseconds // 10**9, nanosec=self.nanoseconds % 10**9)


BUS = Bus()
CLOCK = Clock()
LOG = []


class Publisher:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, message):
        BUS.publish(self.topic, message)

    def get_subscription_count(self):
        return 1


class Node:
    def __init__(self, name):
        self.name = name
        self.timers = []

    def declare_parameter(self, name, value=None):
        value = OVERRIDES.get(self.name, {}).get(name, value)
        return types.SimpleNamespace(value=value)

    def create_publisher(self, msg_type, topic, qos):
        return Publisher(topic)

    def create_subscription(self, msg_type, topic, callback, qos):
        BUS.subs.setdefault(topic, []).append(callback)

    def create_client(self, srv_type, name):
        grid = map_message()
        future = types.SimpleNamespace(result=lambda: types.SimpleNamespace(map=grid))
        return types.SimpleNamespace(wait_for_service=lambda timeout_sec: True,
                                     call_async=lambda request: future)

    def create_timer(self, period, callback):
        self.timers.append((period, callback))

    def get_logger(self):
        return types.SimpleNamespace(info=lambda text: LOG.append((self.name, text)))

    def get_clock(self):
        return CLOCK

    def destroy_node(self):
        pass


def map_message():
    data = np.where(WORLD.occupied, 100, np.where(WORLD.free, 0, -1))
    info = MapMetaData(resolution=WORLD.resolution, width=data.shape[1], height=data.shape[0])
    info.origin.position.x, info.origin.position.y = WORLD.origin[:2]
    return OccupancyGrid(info=info, data=data.ravel().tolist())


class Caster:
    def __init__(self, omap, max_px, *args):
        pass

    def calc_range_repeat_angles(self, queries, angles, out):
        assert queries.dtype == angles.dtype == out.dtype == np.float32
        out[:] = WORLD.raycast(queries, angles).ravel()

    def prune(self):
        pass


class TransformBroadcaster:
    def __init__(self, node):
        pass

    def sendTransform(self, transform):
        BUS.publish('/tf', transform)


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def install(world, repo):
    global WORLD, REPO
    WORLD, REPO = world, repo
    module('rclpy', init=lambda args=None: None, spin=lambda node: None, shutdown=lambda: None,
           spin_until_future_complete=lambda node, future: None)
    module('rclpy.node', Node=Node)
    module('geometry_msgs')
    module('geometry_msgs.msg', Point=Point, Pose=Pose, PoseArray=PoseArray,
           PoseStamped=PoseStamped, PoseWithCovarianceStamped=PoseWithCovarianceStamped,
           PointStamped=PointStamped, Quaternion=Quaternion, TransformStamped=TransformStamped)
    module('nav_msgs')
    module('nav_msgs.msg', Odometry=Odometry)
    module('nav_msgs.srv', GetMap=GetMap)
    module('sensor_msgs')
    module('sensor_msgs.msg', LaserScan=LaserScan)
    module('std_msgs')
    module('std_msgs.msg', Float32=Float32, Float32MultiArray=Float32MultiArray)
    module('tf2_ros', TransformBroadcaster=TransformBroadcaster)
    module('ackermann_msgs')
    module('ackermann_msgs.msg', AckermannDriveStamped=AckermannDriveStamped)
    module('visualization_msgs')
    module('visualization_msgs.msg', Marker=Marker)
    module('ament_index_python')
    module('ament_index_python.packages',
           get_package_share_directory=lambda pkg: '%s/%s' % (REPO, pkg))
    module('range_libc', PyOMap=lambda m: m, PyCDDTCast=Caster, PyRayMarchingGPU=Caster,
           PyRayMarching=Caster, PyBresenhamsLine=Caster)


class LaunchNode:
    def __init__(self, **kw):
        self.kw = kw


class LaunchConfiguration:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context[self.name]


def install_launch():
    """Just enough of launch / launch_ros to call a launch file's setup function."""
    module('launch', LaunchDescription=list)
    module('launch.actions', DeclareLaunchArgument=lambda *a, **kw: None,
           OpaqueFunction=lambda function: function)
    module('launch.substitutions', LaunchConfiguration=LaunchConfiguration)
    module('launch_ros')
    module('launch_ros.actions', Node=LaunchNode)


def reset():
    BUS.subs.clear()
    BUS.log.clear()
    CLOCK.t = 0.0
    LOG.clear()
    OVERRIDES.clear()
