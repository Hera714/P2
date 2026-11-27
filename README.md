# P2 — DINO-WM Reproduction on AutoDL

Reproduction of DINO-WM (goal-conditioned planning with a frozen DINOv2 world model)
on an AutoDL single-GPU server (RTX 4080 SUPER).

## Layout

- `dino_wm/` — project source code (adapted from gaoyuezhou/dino_wm, commit 0a9492fa).
  Local-only adaptations: `conf/plan_pusht_local.yaml` (local Hydra launcher),
  `models/dino.py` (loads DINOv2 from local cache to avoid GitHub network checks).
- `runs/` — task-by-task evaluation logs and artifacts (uploaded selectively;
  large weights / datasets / videos are kept on the server only).

## Notes

Code is a clean snapshot (upstream git history not retained). Original source:
https://github.com/gaoyuezhou/dino_wm
