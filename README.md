# LeHome Challenge — Garment Folding Reproduction

A reproduction study of the [LeHome Challenge 2026](https://lehome-challenge.com/) (ICRA 2026),
a bimanual garment-folding benchmark built on LeRobot SO-101 arms.

This is **not** a competition entry. The goal is to reproduce the official baseline, measure
where it fails, then add the mechanism the winning entry relied on and quantify what it buys.

## Status

`Stage 0 — environment` · nothing trained yet. Numbers below will be filled in as they are measured.

## Why this exists

Garment folding is a long-horizon deformable-object task: a shirt changes shape after every
grasp, so a policy has to perceive state, act, and decide whether it is making progress. The
interesting engineering question is not "which policy scores highest" but "what does a policy do
when it is already failing" — which is exactly what the prizewinning entry attacked.

## Plan

| Stage | Work | Deliverable |
|---|---|---|
| 0 | Isaac Sim 5.1 + Isaac Lab 2.3.1 + LeRobot 0.4.2 environment, dataset download, evaluator running | environment notes, smoke-test rollout |
| 1 | Train the ACT baseline on the released demonstration set | success rate per garment type |
| 2 | Train Diffusion Policy as a second baseline | baseline comparison table |
| 3 | Add self-value prediction: same network predicts action, success probability and task progress; use it for live failure detection and candidate selection | ablation table, one row per component |
| 4 | Small-scale data flywheel: rollouts fed back into training, filtered by completion time and reward | success rate vs number of added trajectories |
| 5 | Failure analysis write-up | short report: what breaks, why, what fixed it |

Stage 1 is the deliverable that matters most — a baseline that actually runs and a number that
can be defended.

## Stack

| Layer | Version |
|---|---|
| Simulator | NVIDIA Isaac Sim 5.1.0 |
| Robot learning framework | Isaac Lab 2.3.1 |
| Policy library | LeRobot 0.4.2 (ACT, Diffusion Policy, SmolVLA) |
| Python | 3.11 |
| Data | `lehome/asset_challenge`, `lehome/dataset_challenge_merged` (HuggingFace, parquet) |
| Hardware target | bimanual SO-101 (leader / follower) |

Reference repositories:

- [lehome-official/lehome-challenge](https://github.com/lehome-official/lehome-challenge) — official environment and evaluator
- [Learning to Fold (arXiv 2606.27163)](https://arxiv.org/abs/2606.27163) — prizewinning solution write-up
- [LeHome simulation environment paper (arXiv 2604.22363)](https://arxiv.org/abs/2604.22363)

## Compute

Local machine: RTX 5070 (12 GB), i7-14700F (28 threads), 32 GB RAM, WSL2.

| Stage | Bottleneck | Feasible locally |
|---|---|---|
| 1–2 baseline training | GPU, ~30k steps | yes |
| 3 self-value heads | GPU, slightly more memory for auxiliary losses | yes |
| 4 flywheel | wall-clock: rollouts are serial and simulation is CPU-bound | yes, at small scale (order 10² trajectories) |
| 5 analysis | none | yes |

The published simulation configuration is CPU-bound, so rollout generation — not training — is
the time sink. That is the same constraint the 17th-place write-up ran into.

Environment note: Isaac Sim 5.1 targets Ubuntu 22.04. This machine's WSL distro is Ubuntu 26.04,
so a dedicated 22.04 distro is needed rather than the existing one.

## Results

| Stage | Metric | Value |
|---|---|---|
| 1 | ACT baseline, success per garment type | _pending_ |
| 2 | Diffusion Policy baseline | _pending_ |
| 3 | + self-value failure detection | _pending_ |
| 3 | + candidate selection | _pending_ |
| 4 | flywheel, 10² trajectories | _pending_ |

## Log

- 2026-11-10 — repository created, plan written, compute budget assessed.

## License

Code released under Apache 2.0, matching the upstream benchmark. Results and text are my own.
