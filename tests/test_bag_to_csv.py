"""scripts/bag_to_csv.py on a bag laid out like a Foxy recording (sqlite3 + metadata.yaml)."""

import os
import pickle
import sqlite3
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def bag_to_csv(monkeypatch):
    # messages are pickled here instead of CDR-serialized, so stand in for the ROS deserializer
    serialization = types.ModuleType('rclpy.serialization')
    serialization.deserialize_message = lambda data, cls: pickle.loads(data)
    utilities = types.ModuleType('rosidl_runtime_py.utilities')
    utilities.get_message = lambda name: None
    monkeypatch.setitem(sys.modules, 'rclpy', types.ModuleType('rclpy'))
    monkeypatch.setitem(sys.modules, 'rclpy.serialization', serialization)
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py', types.ModuleType('rosidl_runtime_py'))
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py.utilities', utilities)
    monkeypatch.syspath_prepend(os.path.join(REPO, 'scripts'))
    sys.modules.pop('bag_to_csv', None)
    import bag_to_csv
    return bag_to_csv


def ns(**fields):
    return types.SimpleNamespace(**fields)


def write_bag(bag, messages):
    os.makedirs(bag)
    open(os.path.join(bag, 'metadata.yaml'), 'w').close()
    db = sqlite3.connect(os.path.join(bag, os.path.basename(bag) + '_0.db3'))
    db.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,'
               ' serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL)')
    db.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,'
               ' timestamp INTEGER NOT NULL, data BLOB NOT NULL)')
    ids = {}
    for topic, type_name, _, _ in messages:
        if topic not in ids:
            ids[topic] = len(ids) + 1
            db.execute('INSERT INTO topics VALUES (?, ?, ?, ?, ?)',
                       (ids[topic], topic, type_name, 'cdr', ''))
    for topic, _, stamp, msg in messages:
        db.execute('INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)',
                   (ids[topic], stamp, pickle.dumps(msg)))
    db.commit()
    db.close()


def read_csv(path):
    with open(path) as f:
        return [line.strip().split(',') for line in f]


def test_replay_bag_to_csv(bag_to_csv, tmp_path, monkeypatch):
    pose = ns(pose=ns(pose=ns(position=ns(x=1.0, y=2.0),
                              orientation=ns(x=0.0, y=0.0, z=0.0, w=1.0))),
              twist=ns(twist=ns(linear=ns(x=0.5))))
    bag = str(tmp_path / 'sim_test_replay_k0.5')
    write_bag(bag, [
        ('/replay/pf/stats', 'std_msgs/msg/Float32MultiArray', 2_500_000_000,
         ns(data=[0.9, 0.1, 0.5])),
        ('/scan', 'sensor_msgs/msg/LaserScan', 1_000_000_000, ns()),
        ('/replay/pf/pose/odom', 'nav_msgs/msg/Odometry', 2_000_000_000, pose),
        ('/replay/pf/confidence', 'std_msgs/msg/Float32', 3_000_000_000, ns(data=0.8)),
    ])
    monkeypatch.setattr(sys, 'argv', ['bag_to_csv.py', bag + '/'])
    bag_to_csv.main()

    out = bag + '_csv'
    assert sorted(os.listdir(out)) == ['confidence.csv', 'pose.csv', 'stats.csv']
    assert read_csv(os.path.join(out, 'pose.csv')) == [
        ['t', 'x', 'y', 'yaw', 'speed'], ['0.0', '1.0', '2.0', '0.0', '0.5']]
    assert read_csv(os.path.join(out, 'stats.csv')) == [
        ['t', 'inlier', 'spread', 'ess'], ['0.5', '0.9', '0.1', '0.5']]
    assert read_csv(os.path.join(out, 'confidence.csv')) == [['t', 'confidence'], ['1.0', '0.8']]


def test_unfinished_bag_is_reported(bag_to_csv, tmp_path):
    bag = str(tmp_path / 'sim_test')
    write_bag(bag, [])
    os.remove(os.path.join(bag, 'metadata.yaml'))
    with pytest.raises(SystemExit, match='not a finished bag'):
        bag_to_csv.read_bag(bag)
