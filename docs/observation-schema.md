# Observation Schema

The current planned RL observation vector is defined in `src/aiget/observation_schema.py` and exposed through the root wrapper `goi_observation_schema.py`.

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
- `progress_features` as current height, best height, and time since last upward progress

Stream the implemented subset:

```bash
python goi_observation_state.py --format json
```

Print the full schema with offsets and sizes:

```bash
python goi_observation_schema.py --format markdown
```
