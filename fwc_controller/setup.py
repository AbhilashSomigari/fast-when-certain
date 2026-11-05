from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'fwc_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*_launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'waypoints'),
            glob('waypoints/*.csv')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Abhilash Somigari',
    maintainer_email='197832085+AbhilashSomigari@users.noreply.github.com',
    description='Pure pursuit with confidence-aware speed, and the '
                'experiment launch file.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = fwc_controller.controller_node:main',
        ],
    },
)
