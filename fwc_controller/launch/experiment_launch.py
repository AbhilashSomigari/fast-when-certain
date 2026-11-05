"""
Launch one experiment configuration: particle filter + controller.

  ros2 launch fwc_controller experiment_launch.py config:=both speed_scale:=0.4

Args:
  config       baseline | noise | speed | both
               noise = speed-scaled motion noise in the filter,
               speed = confidence-aware speed and lookahead in the controller
  speed_scale  multiplies the whole speed profile
  sim          true in the f1tenth_gym sim: odometry from /ego_racecar/odom,
               levine map and waypoints, the sim's own map server
  waypoints    waypoint file in fwc_controller/waypoints (default track.csv,
               levine.csv when sim:=true)
  particles    overrides max_particles (empty = localize.yaml)
  noise_speed_gain
               overrides k of the noise configs (empty = experiment.yaml)
  replay       true = filter + map server only, for scripts/replay.sh; the
               filter's outputs move to /replay/pf/... so they don't mix with
               the /pf/... messages played back from the bag

Values come from fwc_pf/config/localize.yaml, then config/experiment.yaml;
the config arg only switches noise_speed_gain and adaptive_speed on or off.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

CONFIGS = {
    # config: (adaptive noise, adaptive speed)
    'baseline': (False, False),
    'noise': (True, False),
    'speed': (False, True),
    'both': (True, True),
}
REPLAY_TOPICS = ['/pf/pose/odom', '/pf/stats', '/pf/confidence']


def launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    config = arg('config')
    speed_scale = float(arg('speed_scale'))
    sim = arg('sim').lower() == 'true'
    replay = arg('replay').lower() == 'true'
    waypoints = arg('waypoints')
    if config not in CONFIGS:
        raise RuntimeError('config must be one of ' + ', '.join(CONFIGS))
    adaptive_noise, adaptive_speed = CONFIGS[config]
    if not waypoints:
        waypoints = 'levine.csv' if sim else 'track.csv'

    pf_share = get_package_share_directory('fwc_pf')
    localize = os.path.join(pf_share, 'config', 'localize.yaml')
    experiment = os.path.join(
        get_package_share_directory('fwc_controller'),
        'config', 'experiment.yaml')

    pf_overrides = {}
    if not adaptive_noise:
        pf_overrides['noise_speed_gain'] = 0.0
    elif arg('noise_speed_gain'):
        pf_overrides['noise_speed_gain'] = float(arg('noise_speed_gain'))
    if arg('particles'):
        pf_overrides['max_particles'] = int(arg('particles'))
    if sim:
        pf_overrides['odometry_topic'] = '/ego_racecar/odom'

    actions = [
        Node(
            package='fwc_pf',
            executable='particle_filter',
            name='particle_filter',
            output='screen',
            parameters=[localize, experiment, pf_overrides],
            remappings=([(t, '/replay' + t) for t in REPLAY_TOPICS]
                        if replay else []),
        ),
    ]
    if not replay:
        actions.append(Node(
            package='fwc_controller',
            executable='controller',
            name='controller',
            output='screen',
            parameters=[experiment, {
                'speed_scale': speed_scale,
                'adaptive_speed': adaptive_speed,
                'waypoints': waypoints,
            }],
        ))

    # The live sim runs its own map server; everywhere else we start one.
    if replay or not sim:
        with open(localize) as f:
            map_params = yaml.safe_load(f)['map_server']['ros__parameters']
        map_name = 'levine' if sim else map_params['map']
        actions += [
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                parameters=[{'yaml_filename': os.path.join(
                                pf_share, 'maps', map_name + '.yaml')},
                            {'topic': 'map'},
                            {'frame_id': 'map'}],
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_localization',
                output='screen',
                parameters=[{'autostart': True},
                            {'node_names': ['map_server']}],
            ),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='baseline'),
        DeclareLaunchArgument('speed_scale', default_value='0.4'),
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('waypoints', default_value=''),
        DeclareLaunchArgument('particles', default_value=''),
        DeclareLaunchArgument('noise_speed_gain', default_value=''),
        DeclareLaunchArgument('replay', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
