# fwc_pf

Particle filter localization, copied from [f1tenth/particle_filter](https://github.com/f1tenth/particle_filter) (branch `foxy-devel`, MIT license, see `LICENSE`).

Changes from upstream:

- Package renamed `particle_filter` → `fwc_pf`, so it can sit next to an upstream copy in the same workspace. The node and executable are still named `particle_filter`.
- Publishes `/pf/stats` (`std_msgs/Float32MultiArray`, `[spread, ess, match]`) after every update. See `compute_stats()`.
- `config/localize.yaml` loads `maps/track`.
