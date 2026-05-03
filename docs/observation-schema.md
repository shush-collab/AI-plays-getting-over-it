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
- `rich_state_valid_mask`
- `rich_state_source`
- `layout_discovered_at`

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
- `cursor_position_xy`
- `cursor_velocity_xy` derived from successive cursor samples
- `body_position_xy` when startup discovery resolves a raw-readable body position address
- `body_velocity_xy` derived from successive raw body samples when `body_position_xy` is valid
- `body_rotation_sin_cos` when startup discovery resolves a raw-readable body angle
- `body_angular_velocity` derived from successive raw body-angle samples when `body_angle` is valid
- `hammer_anchor_xy` when startup discovery resolves a raw-readable hammer-anchor address
- `hammer_tip_xy` when startup discovery resolves a raw-readable hammer-tip address
- `hammer_direction_sin_cos` derived from raw hammer anchor/tip samples when both are valid
- `hammer_angular_velocity` derived from successive raw hammer-direction samples when both are valid
- `hammer_contact_flags` from raw `HammerCollisions` fields and the managed contacts array
- `hammer_contact_normal_xy` from the first raw `ContactPoint2D` record when available
- `progress_features` from raw height plus local progress tracking when body height is valid
- `previous_action` echoed from the CLI input

Current observation architecture:

- Fast lane:
  - `cursor_position_xy`
  - `cursor_velocity_xy`
- Startup discovery:
  - ptrace/IL2CPP authoritative sampling
  - raw rich-layout resolution
  - cache-first layout reuse
  - optional one-shot layout validation
- Raw-memory live lane:
  - body state
  - hammer state
  - hammer contact state
  - progress features
- The fast loop reuses the latest rich snapshot instead of blocking on a fresh raw rich-state update.
- Fields that cannot be resolved to raw memory stay zero/default in the payload and are marked `false` in `rich_state_valid_mask`.
- The default startup path resolves must-have rich fields first and leaves optional fields invalid rather than blocking the stream.

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
python goi_observation_state.py --format text --samples 0 --interval 0.05 --rich-snapshot-interval 0.2
```

Resolve and validate the raw rich layout once:

```bash
python goi_observation_state.py --format json --validate-layout --live-layout-cache /tmp/goi-layout.json
```

Fast partial startup:

```bash
python goi_observation_state.py --format json --layout-discovery-timeout 1.5
```

Print the full schema with offsets and sizes:

```bash
python goi_observation_schema.py --format markdown
```
