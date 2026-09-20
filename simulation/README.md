# AIget standalone simulation

This is the first standalone game-engine milestone: a C17, Box2D, and raylib
implementation of the owned game's core hammer rig and collision map. It does
not load Unity at runtime.

## What is represented

- Player, hub, slider, handle, pole, and tip are separate Box2D bodies.
- Masses, drag, joint limits, motor-force ceilings, and weld tuning were read
  from the owned Unity scene.
- The player's measured PolygonCollider2D outline is used.
- `aiget_export_level` reads the owned Unity 2020.3.6f1 scene and writes a
  plain collision-map file. The standalone engine can then run it without Unity.
- `aiget_game` renders the physics world with raylib; mouse deltas drive the
  hammer target, `R` resets, mouse wheel zooms, `H` toggles help, and `Esc`
  quits.

## Build and test

```sh
cmake -S simulation -B build/simulation-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/simulation-release --parallel
ctest --test-dir build/simulation-release --output-on-failure
```

The first configure fetches Box2D 3.1.1 and the raylib 5.5 Linux x86_64
distribution. For a headless simulation/test-only build, configure with
`-DAIGET_VIEWER=OFF`.

## Export and play an owned map

```sh
./build/simulation-release/aiget_export_level \
  /path/to/GettingOverIt_Data/level1 \
  /path/to/GettingOverIt_Data/sharedassets1.assets \
  runs/simulation/owned-level.txt

./build/simulation-release/aiget_sim_test runs/simulation/owned-level.txt
./build/simulation-release/aiget_game --level runs/simulation/owned-level.txt
```

The exporter is intentionally fail-closed: it accepts only the verified
Unity 2020.3.6f1 serialized layout and never overwrites an existing output.
It can write a usable partial map while exiting nonzero if it finds a collider
with behavior it does not implement.

## Deliberate next fidelity work

This is an engine foundation, not a tick-perfect Unity clone yet:

- Box2D and Unity Physics2D use different solvers.
- The mouse-to-motor controller is a provisional C controller; matching the
  live `PlayerControl` path and feeding the same timestamped input stream is
  the next validation milestone.
- The hammer tip uses a validated convex reduction of the measured Unity
  outline. Its sub-5-mm concave detail cannot be represented directly by
  Box2D's valid-polygon tolerance.
- The owned map export currently reports and omits `PlatformEffector2D`
  behavior (the source scene's `OuterRock` collider). Other imported terrain
  uses its exact outline, represented as one-sided Box2D chains.
