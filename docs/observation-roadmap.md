# Observation Roadmap

This roadmap focuses on making live observation safe enough for continuous RL use in
*Getting Over It with Bennett Foddy* on native Linux.

## Current State

- The fast cursor signal is validated and can be streamed continuously from raw memory.
- Rich live streaming no longer calls ptrace or IL2CPP remote execution after startup discovery.
- The rich observation path now runs as a slower raw-memory background lane driven by a reusable `ResolvedLiveLayout`.
- Startup is now cache-first, time-bounded, and allowed to fall back to a partial rich layout.
- The live payload includes rich-state metadata such as `rich_state_ts`, `rich_state_age`, `rich_state_valid_mask`, and `rich_state_source`.
- The current bottleneck is observation architecture, not data discovery.

## Goal

Move from scattered live reads toward one controlled observation pipeline:

1. keep the fast cursor signal on the high-frequency path
2. move richer state to a slower snapshot path
3. unify policy input behind one observation builder
4. later replace external rich-state reads with one exported in-game snapshot

## Phase 1: Make The Current System Playable

- [x] Task 1: Freeze the fast cursor lane as the stable high-frequency observation source.
- [x] Task 2: Split observation into explicit fast and slow lanes.
- [x] Task 3: Batch slow-lane reads into one `RichStateSnapshot` object.
- [x] Task 4: Allow the policy loop to reuse the latest rich snapshot without blocking.
- [x] Remove remote rich live streaming from the default path.
- [x] Add startup-only rich raw-layout discovery and optional validation.
- [x] Make rich-layout startup cache-first and time-bounded with partial fallback.

## Phase 2: Clean Observation Interface

- [ ] Task 5: Create one observation builder API that merges fast and slow state.
- [ ] Task 6: Add metrics for observer rate, rich-snapshot rate, playability, and overhead.

## Phase 3: Replace External Rich Reads

- [ ] Task 7: Move rich-state assembly onto the game side.
- [ ] Task 8: Expose one stable exported snapshot to the RL process.
- [ ] Task 9: Keep or retire the separate fast cursor lane based on exporter responsiveness.

## Definition Of Success

- The game remains playable while observation is running.
- The policy loop receives continuous observations.
- Rich state is available often enough for training/live control.
- Observation code is modular enough to swap backends without rewriting the RL loop.

## Status

Phase 1 is complete.

- The fast cursor lane is frozen and reusable.
- The slow rich-state lane is explicit, batched, raw-memory only, and non-blocking from the fast loop's point of view.
- Split lanes are done.
- Remote rich live streaming is removed.
- Raw rich live layout discovery is added.
- Cache-first partial startup is added.
- The remaining major architectural work is the unified observation API plus eventual in-game rich-state export.
