"""Unit tests for fwc_pf/fwc_pf/mcl.py."""

import numpy as np
import pytest

import sim2d
from sim2d import mcl

START = np.array([3.09, -0.51, 0.0])   # class raceline start, facing +x


@pytest.fixture(scope='module')
def world():
    return sim2d.World('track')


def make_pf(world, n=500, seed=0, **params):
    return mcl.ParticleFilter(world.grid, world.raycast, rng=np.random.default_rng(seed),
                              max_particles=n, **params)


def upstream_table(max_px, z_hit, z_short, z_max, z_rand, sigma):
    # the loop from upstream precompute_sensor_model, as the reference
    n = max_px + 1
    table = np.zeros((n, n))
    for d in range(n):
        norm = 0.0
        for r in range(n):
            z = float(r - d)
            prob = z_hit * np.exp(-(z * z) / (2.0 * sigma * sigma)) / (sigma * np.sqrt(2.0 * np.pi))
            if r < d:
                prob += 2.0 * z_short * (d - r) / float(d)
            if r == max_px:
                prob += z_max
            if r < max_px:
                prob += z_rand / float(max_px)
            norm += prob
            table[r, d] = prob
        table[:, d] /= norm
    return table


def test_sensor_table_matches_upstream():
    args = (40, 0.75, 0.01, 0.07, 0.12, 8.0)
    assert np.allclose(mcl.sensor_table(*args), upstream_table(*args))


def test_odom_delta_in_previous_frame_and_wrapped():
    # facing +y, moved 1 m along +y and turned left 0.1 rad: 1 m forward
    d = mcl.odom_delta(np.array([0.0, 0.0, np.pi / 2]), np.array([0.0, 1.0, np.pi / 2 + 0.1]))
    assert np.allclose(d, [1.0, 0.0, 0.1])
    # crossing +-pi is a small turn, not -2 pi
    d = mcl.odom_delta(np.array([0.0, 0.0, 3.1]), np.array([0.0, 0.0, -3.1]))
    assert abs(d[2] - (2 * np.pi - 6.2)) < 1e-9


def test_heading_mean_is_circular(world):
    pf = make_pf(world, n=2000)
    pf.particles = np.column_stack([np.full(2000, 3.0), np.full(2000, -0.5), np.concatenate(
        [np.full(1000, 0.05), np.full(1000, 2 * np.pi - 0.05)])])
    assert abs(pf.expected_pose()[2]) < 1e-9


def test_init_pose_and_global(world):
    pf = make_pf(world, n=4000)
    assert not pf.initialized
    pf.init_pose(*START)
    assert pf.initialized
    assert np.allclose(pf.particles.mean(axis=0), START, atol=0.05)
    assert np.allclose(pf.particles.std(axis=0), [0.5, 0.5, 0.4], atol=0.05)
    pf.init_global()
    cols = np.floor((pf.particles[:, 0] - world.origin[0]) / world.resolution).astype(int)
    rows = np.floor((pf.particles[:, 1] - world.origin[1]) / world.resolution).astype(int)
    assert world.free[rows, cols].all()
    assert pf.particles[:, 2].std() > 1.5    # headings all around


def test_motion_model_mean_and_speed_scaled_noise(world):
    for k, speed in ((0.0, 3.0), (0.5, 3.0)):
        pf = make_pf(world, n=20000, noise_speed_gain=k)
        pf.particles = np.tile([0.0, 0.0, np.pi / 2], (20000, 1))
        pf.motion_model(np.array([0.2, 0.0, 0.0]), speed)
        # 0.2 m forward while facing +y
        assert np.allclose(pf.particles[:, :2].mean(axis=0), [0.0, 0.2], atol=0.01)
        scale = 1.0 + k * speed
        # noise is in the map frame, as upstream
        assert pf.particles[:, 0].std() == pytest.approx(0.05 * scale, rel=0.05)
        assert pf.particles[:, 1].std() == pytest.approx(0.025 * scale, rel=0.05)
        assert pf.particles[:, 2].std() == pytest.approx(0.25 * scale, rel=0.05)


def test_sensor_model_peaks_at_true_pose(world):
    obs = world.raycast(START[None, :], sim2d.ANGLES)[0]
    pf = make_pf(world, n=441)
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))
    pf.particles = np.column_stack([START[0] + gx.ravel(), START[1] + gy.ravel(), np.zeros(441)])
    pf.sensor_model(obs, sim2d.ANGLES)
    best = pf.particles[np.argmax(pf.weights)]
    assert np.hypot(*(best[:2] - START[:2])) < 0.08
    assert pf.weights.sum() == pytest.approx(1.0)


def test_update_survives_a_scan_that_fits_nowhere(world):
    pf = make_pf(world)
    pf.init_pose(*START)
    for obs in (np.zeros(len(sim2d.ANGLES)), np.full(len(sim2d.ANGLES), np.inf),
                np.full(len(sim2d.ANGLES), np.nan)):
        pf.update(np.zeros(3), 0.0, obs, sim2d.ANGLES)
        assert np.isfinite(pf.weights).all() and pf.weights.sum() == pytest.approx(1.0)
        assert np.isfinite(pf.pose).all() and np.isfinite(pf.stats).all()


def test_inlier_reads_the_pose_not_the_cloud(world):
    obs = world.raycast(START[None, :], sim2d.ANGLES)[0]
    pf = make_pf(world, n=1000)
    for spread in (0.01, 0.3):
        pf.particles = START + np.random.default_rng(1).normal(0, spread, (1000, 3))
        pf.weights[:] = 1.0 / 1000
        pf.pose = START.copy()
        inlier, s, ess = pf.compute_stats(obs, sim2d.ANGLES)
        assert inlier == 1.0
        assert s == pytest.approx(spread * np.sqrt(2), rel=0.1)
    pf.pose = START + [0.0, 0.3, 0.3]           # wrong pose: the scan no longer fits
    assert pf.compute_stats(obs, sim2d.ANGLES)[0] < 0.5


def test_confidence_mapping_smoothing_reset():
    c = mcl.Confidence(0.7, 0.95, 0.5)
    assert c.value == 0.0
    assert c.update(1.0) == pytest.approx(0.5)       # halfway to 1
    assert c.update(1.0) == pytest.approx(0.75)
    c.reset()
    assert c.value == 0.0
    c.alpha = 1.0
    assert c.update(0.825) == pytest.approx(0.5)     # linear between low and high
    assert c.update(0.2) == 0.0 and c.update(2.0) == 1.0
    with pytest.raises(ValueError):
        mcl.Confidence(0.5, 0.5, 0.3)


def test_unknown_parameter_is_rejected(world):
    with pytest.raises(ValueError):
        mcl.ParticleFilter(world.grid, world.raycast, conf_low=0.5)
