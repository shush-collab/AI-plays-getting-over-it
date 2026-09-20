# Physics-only training worker

`runs/game-backups/getting-over-it-build-8111718-original` is the immutable local backup of the owned game build. `runs/training-game/getting-over-it-physics-worker` is the only copy this tooling changes.

The worker keeps the original scene's collision map, `Rigidbody2D` bodies, 2D joints, physics materials, controller scripts, animators, and cameras. Its asset patch disables presentation components in `level1` and, only when explicitly requested, detaches scene references to render meshes/materials/sprites and audio clips. Its BepInEx plugin repeats the component reduction at runtime for objects instantiated after scene load and asks Unity to unload unreferenced assets. It never modifies colliders, rigidbodies, joints, or gameplay scripts.

Rebuild the editable worker scene from the immutable local backup, then apply the aggressive presentation-reference reduction:

```bash
aiget-physics-worker \
  --game-root runs/training-game/getting-over-it-physics-worker \
  --backup-root runs/game-backups/getting-over-it-build-8111718-original \
  --apply --detach-presentation-assets --rebuild-from-backup \
  --manifest runs/training-game/getting-over-it-physics-worker/physics-worker-manifest.json
```

Verify the preserved components against the backup before running training:

```bash
aiget-verify-physics-worker \
  --game-root runs/training-game/getting-over-it-physics-worker \
  --backup-root runs/game-backups/getting-over-it-build-8111718-original
```

Build and install the runtime plugin into a worker copy:

```bash
scripts/game/build_physics_worker_plugin.sh
```

Pass a different worker copy as the first argument when creating additional instances. The build machine needs a .NET 8 SDK; the game only needs the deployed DLL plus its installed BepInEx IL2CPP runtime.

Run a worker headlessly through BepInEx:

```bash
cd runs/training-game/getting-over-it-physics-worker
./run_bepinex.sh GettingOverIt.x86_64 -batchmode -nographics -logFile worker.log
```

After its first startup pass, the plugin writes `BepInEx/config/AIgetPhysicsWorker/status.json`. This reports what presentation components it disabled, whether it requested unused-asset collection, and the process working set; it does not establish physics equivalence. The next gate is a deterministic reset/action/contact comparison against the untouched backup before training or changing solver settings.
