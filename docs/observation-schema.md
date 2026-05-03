# Observation Schema

The v1 training observation is a Gymnasium Dict. The vector part is defined by
`src/aiget/observation_schema.py` and filled by `src/aiget/observation_vector.py`.

Default settings:
- schema version: `v1`
- body rays: `0`
- hammer rays: `0`
- action dimension: `2`
- state vector dimension: `32`
- image shape: `(84, 84, 1)`
- image dtype/range: `uint8`, `[0, 255]`

Current groups in order:
- `cursor_position_xy`
- `cursor_velocity_xy`
- `body_position_xy`
- `body_velocity_xy`
- `hammer_tip_xy`
- `hammer_direction_sin_cos`
- `progress_features`
- `previous_action`
- `valid_mask`

V1 live readers:
- Fast lane every environment step:
  - `cursor_position_xy`
  - `cursor_velocity_xy`
- Rich lane at lower rate:
  - `body_position_xy`
  - `body_velocity_xy`
  - `hammer_tip_xy`
  - `hammer_direction_sin_cos`
  - `progress_features`

The valid mask covers the 15 state values before `previous_action`:
- cursor x/y/vx/vy
- body x/y/vx/vy
- hammer tip x/y
- hammer direction sin/cos
- progress y, best progress y, time since progress

Missing rich fields are emitted as `0.0` and their mask values are `0.0`.
Cursor fields are mask `1.0` when the fast raw-memory read succeeds.

Removed from v1 training observations:
- body contacts
- contact normals
- hammer contact object pointer chasing
- LIDAR/raycast features
- body angle and angular velocity
- hammer angular velocity

Runtime architecture:
- Startup may use ptrace/IL2CPP to resolve raw addresses.
- Runtime play uses fixed-address raw memory reads only.
- The fast lane reads `fakeCursorRB_native + 0xA8` directly.
- The rich lane reads batched raw address groups in a background thread.
- The image lane captures low-resolution grayscale frames in a background thread.
- `GettingOverItEnv.step()` never waits for a fresh rich snapshot.
- `GettingOverItEnv.step()` also does not capture screen frames synchronously.

Print the schema:

```bash
python goi_observation_schema.py --format markdown
```

Use the Gymnasium environment:

```python
from aiget.env import GettingOverItEnv

env = GettingOverItEnv(dt=1.0 / 30.0)
obs, info = env.reset()
state = obs["state"]
image = obs["image"]
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

For SB3, use `MultiInputPolicy`, not `MlpPolicy`, because observations include
both image and vector state.
