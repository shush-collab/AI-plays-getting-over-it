# Observation Schema

The current planned RL observation vector is defined in `src/aiget/observation_schema.py` and exposed through the root wrapper `goi_observation_schema.py`.

This file describes the planned policy observation layout.

The live JSON stream from `goi_observation_state.py` also includes extra transport metadata such as:

- `ts`
- `pid`
- `addr`
- `rich_state_ts`
- `rich_state_age`
- `rich_state_valid`

Default settings:
- schema version: `v1`
- body rays: `32`
- hammer rays: `32`
- action dimension: `2`
- flat observation dimension: `95`

Current groups in order:
- body position
- body velocity
- body rotation as `sin/cos`
- body angular velocity
- hammer anchor position
- hammer tip position
- hammer direction as `sin/cos`
- hammer angular velocity
- `fakeCursorRB` position
- `fakeCursorRB` velocity
- body contact flags
- body contact normal
- hammer contact flags
- hammer contact normal
- progress features
- previous action
- body LIDAR distances
- hammer LIDAR distances

Currently implemented live readers:
- `body_position_xy` from `PoseControl.potMeshHub`
- `body_velocity_xy` derived from successive body samples
- `body_rotation_sin_cos` from the live `potMeshHub` transform quaternion
- `body_angular_velocity` derived from successive body rotation samples
- `hammer_anchor_xy` from `PoseControl.handle`
- `hammer_tip_xy` from `PoseControl.tip`
- `hammer_direction_sin_cos` derived from `handle -> tip`
- `hammer_angular_velocity` derived from successive hammer direction samples
- `cursor_position_xy`
- `cursor_velocity_xy` derived from successive cursor samples
- `hammer_contact_flags` from `HammerCollisions.contacts` and `HammerCollisions.slide`
- `hammer_contact_normal_xy` inferred from the first `ContactPoint2D` record in `HammerCollisions.contacts`
- `progress_features` as current height, best height, and time since last upward progress
- `previous_action` echoed from the CLI input

Current observation architecture:

- Fast lane:
  - `cursor_position_xy`
  - `cursor_velocity_xy`
- Slow rich lane:
  - body state
  - hammer state
  - hammer contact state
  - progress features
- The fast loop reuses the latest rich snapshot instead of blocking on a fresh slow update.

Still missing from the current live stream:
- `body_contact_flags`
- `body_contact_normal_xy`
- synthetic body LIDAR distances
- synthetic hammer LIDAR distances

One-shot rich observation payload:

```bash
python goi_observation_state.py --format json
```

Continuous split-lane observation stream:

```bash
python goi_observation_state.py --format text --samples 0 --interval 0.05 --unity-snapshot-interval 0.2
```

Print the full schema with offsets and sizes:

```bash
python goi_observation_schema.py --format markdown
```
