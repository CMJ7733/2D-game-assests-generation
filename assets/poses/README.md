# Pose Skeleton Library

12 OpenPose-format skeleton PNGs at 512×512, used as ControlNet conditioning input
to enforce character pose across animation frames.

- `idle_0..3.png` — 4-frame idle cycle (subtle weight shift)
- `walk_0..7.png` — 8-frame side-view walk cycle

Source: hand-curated via `scripts/prepare_poses.py` applied to reference frames.
