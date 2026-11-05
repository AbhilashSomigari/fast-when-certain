# Run log

## Failure rule (fixed before any data; do not change)

A lap fails if any of these happen during it:

- a takeover: LB pressed while the car is driving itself (`/teleop` while moving)
- the filter pose jumps more than 0.5 m between two messages
- the car touches a wall (not in the bag: count it in Notes as `wall`)

`analysis/summarize.py` finds the first two. Hand driving that starts from standstill, such as
driving back onto the start mark after the last lap, is not a takeover.

## Runs

One row per bag, filled in the same day. Bag names: `<config>_s<speed_scale>`, e.g.
`baseline_s0.5` or `final_both_s0.6`.

| Date | Bag | Config | speed_scale | Laps | Failures | Takeovers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

## Settings found

| What | Value | From |
| --- | --- | --- |
| Test speed S* (speed_scale) |  | baseline sweep, session 1 |
| max_particles | 2000 | 1000 if the baseline never failed |
| confidence_signal, conf_low, conf_high |  | summarize.py --signals |
| conf_min_speed |  | sweep |
| noise_speed_gain (k) |  | replay.sh, or 0.5 if replay didn't work |
