# Run log

## Failure rule (fixed before any data; do not change)

A lap fails if any of these happen during it:

- a takeover: LB pressed while the car is driving itself (`/teleop` while moving)
- the filter pose jumps more than 0.5 m between two messages
- the car touches a wall (not in the bag: count it in Notes as `wall`)

`analysis/summarize.py` finds the first two. Hand driving that starts from standstill, such as
driving to the markers after the last lap, is not a takeover.

## Runs

One row per bag, filled in the same day. Bag names: `<config>_s<speed_scale>`, e.g.
`baseline_s0.5` or `final_both_s0.6`.

| Date | Bag | Config | speed_scale | Laps | Failures | Takeovers | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

## Floor markers

Map-frame positions, read from `/pf/pose/odom` after parking 10 s on each marker.

| Marker | Where | x (m) | y (m) |
| --- | --- | --- | --- |
| M1 | straight |  |  |
| M2 | straight |  |  |
| M3 | corner |  |  |
| M4 | corner |  |  |

## Calibration

| Parameter | Before | After |
| --- | --- | --- |
| speed_to_erpm_gain |  |  |
| steering_angle_to_servo_offset |  |  |
| steering_angle_to_servo_gain |  |  |
