"""The ROS nodes and the launch file, run on fake ROS (tests/fake_ros.py) against the 2D sim."""

import importlib
import os
import sys

import numpy as np
import pytest

import fake_ros
import sim2d

WORLD = sim2d.World('track')
fake_ros.install(WORLD, sim2d.REPO)
fake_ros.install_launch()
from fwc_pf import particle_filter  # noqa: E402
from fwc_controller import controller_node  # noqa: E402

sys.path.insert(0, os.path.join(sim2d.REPO, 'fwc_controller', 'launch'))
experiment_launch = importlib.import_module('experiment_launch')

FULL_ANGLES = np.linspace(-2.35, 2.35, 1080)
LAUNCH_DEFAULTS = {'config': 'baseline', 'speed_scale': '0.6', 'sim': 'false', 'replay': 'false',
                   'waypoints': '', 'particles': '', 'noise_speed_gain': ''}


@pytest.fixture(autouse=True)
def clean_bus():
    fake_ros.reset()
    yield


def launch(**args):
    """The launch file's nodes as {name: merged parameter dict}."""
    nodes = experiment_launch.launch_setup(dict(LAUNCH_DEFAULTS, **args))
    out = {}
    for node in nodes:
        params = {}
        for p in node.kw.get('parameters', []):
            params.update(p)
        out[node.kw.get('name')] = (params, node.kw)
    return out


def test_every_launch_parameter_is_declared():
    for config in ('baseline', 'noise', 'speed', 'both'):
        for sim in ('false', 'true'):
            nodes = launch(config=config, sim=sim, particles='800', noise_speed_gain='0.3')
            assert set(nodes['particle_filter'][0]) <= set(particle_filter.PARAMS)
            assert set(nodes['controller'][0]) <= set(controller_node.PARAMS)


def test_configs_switch_noise_and_speed():
    want = {'baseline': (0.0, False), 'noise': (0.5, False), 'speed': (0.0, True),
            'both': (0.5, True)}
    for config, (k, adaptive) in want.items():
        nodes = launch(config=config)
        assert nodes['particle_filter'][0]['noise_speed_gain'] == k
        assert nodes['controller'][0]['adaptive_speed'] is adaptive
        assert nodes['controller'][0]['speed_scale'] == 0.6
    # launch overrides beat the yaml files
    nodes = launch(config='both', particles='800', noise_speed_gain='0.25', sim='true')
    pf = nodes['particle_filter'][0]
    assert pf['max_particles'] == 800 and pf['noise_speed_gain'] == 0.25
    assert pf['odometry_topic'] == '/ego_racecar/odom'
    assert nodes['controller'][0]['waypoints'] == 'levine.csv'
    assert launch(config='baseline', noise_speed_gain='0.25')['particle_filter'][0][
        'noise_speed_gain'] == 0.0


def test_replay_runs_the_filter_alone_on_replay_topics():
    nodes = launch(config='noise', replay='true', noise_speed_gain='1.0')
    assert 'controller' not in nodes and 'map_server' in nodes
    params, kw = nodes['particle_filter']
    assert params['noise_speed_gain'] == 1.0
    assert ('/pf/stats', '/replay/pf/stats') in kw['remappings']


def drive(launch_args, seconds, init_pose=True, rate=40.0):
    """Run the launch file's two nodes on the bus with a simulated car. Returns the car pose."""
    nodes = launch(**launch_args)
    for name in ('particle_filter', 'controller'):
        fake_ros.OVERRIDES[name] = nodes[name][0]
    pf_node = particle_filter.ParticleFilterNode()
    ctl_node = controller_node.Controller()
    command = {'speed': 0.0, 'steer': 0.0}

    def on_drive(msg):
        command['speed'], command['steer'] = msg.drive.speed, msg.drive.steering_angle
    fake_ros.BUS.subs.setdefault('/drive', []).append(on_drive)

    path = ctl_node.pursuit.path
    heading = np.arctan2(*(path[5] - path[0])[::-1])
    car = np.array([path[0, 0], path[0, 1], heading])
    v, dt = 0.0, 1.0 / rate
    for step in range(int(seconds * rate)):
        fake_ros.CLOCK.t = step * dt
        if step == 5 and init_pose:
            msg = fake_ros.PoseWithCovarianceStamped()
            msg.pose.pose.position.x, msg.pose.pose.position.y = car[0] + 0.2, car[1] - 0.1
            msg.pose.pose.orientation = particle_filter.yaw_to_quaternion(car[2] + 0.1)
            fake_ros.BUS.publish('/initialpose', msg)
        scan = fake_ros.LaserScan(angle_min=-2.35, angle_max=2.35,
                                  angle_increment=4.7 / 1079, range_max=30.0)
        scan.ranges = list(WORLD.raycast(car[None, :], FULL_ANGLES)[0])
        fake_ros.BUS.publish('/scan', scan)
        odom = fake_ros.Odometry()
        odom.header.stamp = fake_ros.CLOCK.now().to_msg()
        odom.pose.pose.position.x, odom.pose.pose.position.y = car[0], car[1]
        odom.pose.pose.orientation = particle_filter.yaw_to_quaternion(car[2])
        odom.twist.twist.linear.x = v
        fake_ros.BUS.publish('/odom', odom)
        v += np.clip(command['speed'] - v, -9.5 * dt, 7.0 * dt)
        car = car + [v * np.cos(car[2]) * dt, v * np.sin(car[2]) * dt,
                     v * np.tan(command['steer']) / 0.3302 * dt]
    return car, pf_node, ctl_node


def test_nothing_drives_before_a_2d_pose_estimate():
    drive({'config': 'both'}, 1.0, init_pose=False)
    assert '/pf/pose/odom' not in fake_ros.BUS.log
    assert '/drive' not in fake_ros.BUS.log


@pytest.mark.parametrize('config', ['baseline', 'both'])
def test_nodes_drive_the_car_along_the_raceline(config):
    car, pf_node, ctl_node = drive({'config': config}, 4.0)
    log = fake_ros.BUS.log
    pose = log['/pf/pose/odom'][-1].pose.pose.position
    assert np.hypot(pose.x - car[0], pose.y - car[1]) < 0.15
    idx = ctl_node.pursuit.index
    assert idx * 0.05 > 3.0                                   # got going along the line
    assert np.hypot(*(ctl_node.pursuit.path[idx] - car[:2])) < 0.3
    assert len(log['/pf/stats'][-1].data) == 3
    assert log['/pf/confidence'][-1].data > 0.9
    assert log['/tf'][-1].child_frame_id == 'pf_laser'
    assert log['/pf/viz/scan'][-1].header.frame_id == 'pf_laser'
    assert any('iters per sec' in text for _, text in fake_ros.LOG)
