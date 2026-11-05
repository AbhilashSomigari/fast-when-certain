"""
Launch one experiment configuration: particle filter + controller.

  ros2 launch fwc_controller experiment_launch.py config:=both speed_scale:=0.4

Args:
  config       baseline | noise | speed | both
               noise = speed-scaled motion noise in the filter,
               speed = confidence-aware speed and lookahead in the controller
  speed_scale  multiplies the whole speed profile
  sim          true in the f1tenth_gym sim: odometry from /ego_racecar/odom,
               levine waypoints, the sim's own map server, CPU ray casting
  waypoints    waypoint file in fwc_controller/waypoints (default track.csv,
               levine.csv when sim:=true)

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


def launch_setup(context):
    config = LaunchConfiguration('config').perform(context)
    speed_scale = float(LaunchConfiguration('speed_scale').perform(context))
    sim = LaunchConfiguration('sim').perform(context).lower() == 'true'
    waypoints = LaunchConfiguration('waypoints').perform(context)
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
    if sim:
        # the VM has no GPU: CPU ray casting with fewer particles
        pf_overrides.update({'odometry_topic': '/ego_racecar/odom',
                             'range_method': 'cddt', 'max_particles': 2000})

    actions = [
        Node(
            package='fwc_pf',
            executable='particle_filter',
            name='particle_filter',
            output='screen',
            parameters=[localize, experiment, pf_overrides],
        ),
        Node(
            package='fwc_controller',
            executable='controller',
            name='controller',
            output='screen',
            parameters=[experiment, {
                'speed_scale': speed_scale,
                'adaptive_speed': adaptive_speed,
                'waypoints': waypoints,
            }],
        ),
    ]

    # On the car the map comes from our map server; the sim runs its own.
    if not sim:
        with open(localize) as f:
            map_params = yaml.safe_load(f)['map_server']['ros__parameters']
        map_name = map_params['map']
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
        OpaqueFunction(function=launch_setup),
    ])
