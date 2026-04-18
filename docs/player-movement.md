# Player Movement Notes

## What Was Confirmed

- `PlayerControl.oldHammerPos` is not suitable for live tracking.
- The correct live object is `PlayerControl.fakeCursorRB`.
- `fakeCursorRB` is a `UnityEngine.Rigidbody2D`.
- Unity's own `Rigidbody2D.position` getter was called on the live object to verify the coordinates.

## What The Tracker Reads

- The validated coordinate is the live `Rigidbody2D.position` of `PlayerControl.fakeCursorRB`.
- This is currently the best confirmed movement signal for the player.

## Memory Path

- The runtime resolver finds `PlayerControl`, then `fakeCursorRB`, then the native Unity object behind `fakeCursorRB`.
- After calibration, the direct raw-memory read used by the tracker is `fakeCursorRB_native + 0xA8`.

## Why This Is Trusted

- The raw-memory value was matched against Unity's own `Rigidbody2D.position` result.
- The values matched exactly during live checks.
- The raw-memory stream changed smoothly while the player moved.
