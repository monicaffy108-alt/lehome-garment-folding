# LeHome Challenge — Garment Folding Reproduction

A reproduction study of the [LeHome Challenge 2026](https://lehome-challenge.com/) (ICRA 2026),
a bimanual garment-folding benchmark built on LeRobot SO-101 arms.

This is **not** a competition entry. The goal is to reproduce the official baseline, measure
where it fails, then add the mechanism the winning entry relied on and quantify what it buys.

## Status

**Stage 0 + Stage 1 complete** (2026-11-12). The ACT baseline is trained and evaluated:
**top_long = 83.3% / 75.0%** over two independent 12-episode evaluation runs (random-policy
baseline: 0.0%). Numbers, protocol and every environment pitfall hit on the way are recorded
below. Trained checkpoint and eval-rollout videos are attached to
[Release `v0.1.0-stage1`](https://github.com/monicaffy108-alt/lehome-garment-folding/releases/tag/v0.1.0-stage1).

## Why this exists

Garment folding is a long-horizon deformable-object task: a shirt changes shape after every
grasp, so a policy has to perceive state, act, and decide whether it is making progress. The
interesting engineering question is not "which policy scores highest" but "what does a policy do
when it is already failing" — which is exactly what the prizewinning entry attacked.

## Plan

| Stage | Work | Deliverable |
|---|---|---|
| 0 | Isaac Sim 5.1 + Isaac Lab 2.3.1 + LeRobot 0.4.3 environment, dataset download, evaluator running | environment notes, smoke-test rollout |
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
| Simulator | NVIDIA Isaac Sim 5.1.0 (pip install) |
| Robot learning framework | Isaac Lab 2.3.1 (lehome fork, `isaaclab` 0.53.1) |
| Policy library | LeRobot 0.4.3 (ACT, Diffusion Policy, SmolVLA) |
| PyTorch | 2.7.0+cu128 (Blackwell sm_120 build) |
| Python | 3.11 |
| Data | `top_long_merged` — 250 episodes / 83,068 frames, LeRobot dataset v3.0; `Assets` 946 MB |
| Hardware target | bimanual SO-101 (leader / follower) |

Reference repositories:

- [lehome-official/lehome-challenge](https://github.com/lehome-official/lehome-challenge) — official environment and evaluator
- [Learning to Fold (arXiv 2606.27163)](https://arxiv.org/abs/2606.27163) — prizewinning solution write-up
- [LeHome simulation environment paper (arXiv 2604.22363)](https://arxiv.org/abs/2604.22363)

## Compute

Local machine: RTX 5070 (12 GB), i7-14700F (28 threads), 32 GB RAM, native Ubuntu 24.04.

| Stage | Bottleneck | Feasible locally |
|---|---|---|
| 1–2 baseline training | GPU, ~30k steps | yes |
| 3 self-value heads | GPU, slightly more memory for auxiliary losses | yes |
| 4 flywheel | wall-clock: rollouts are serial and simulation is CPU-bound | yes, at small scale (order 10² trajectories) |
| 5 analysis | none | yes |

The published simulation configuration is CPU-bound, so rollout generation — not training — is
the time sink. That is the same constraint the 17th-place write-up ran into.

Environment note: the official docs target Ubuntu 22.04, but this reproduction ran on **native
Ubuntu 24.04 (kernel 7.0.0-42)** — no dedicated distro needed. The pip-installed Isaac Sim 5.1.0
works unmodified.

## Results

Evaluator protocol: one run = 12 episodes over the garment-variant suite
(10 `Seen` textures + 2 `Unseen` textures not present in the training set).
`--num_episodes N` means N episodes **per variant**. Success is a final-state check,
not the dense return — failed episodes frequently show *higher* return (more folding
progress) while missing the success criterion.

| Stage | Metric | Value |
|---|---|---|
| 1 | Random policy baseline (12 episodes) | 0.00% success, return 106.9 ± 2.1 |
| 1 | **ACT baseline, top_long — run 1** (12 episodes) | **83.33% success** (10/12), return 115.4 ± 23.1 |
| 1 | **ACT baseline, top_long — run 2** (12 episodes) | **75.00% success** (9/12), return 124.2 ± 30.6 |
| 1 | ACT, generalization to Unseen variants | run 1: 2/2 succeeded; full per-variant tables in console logs |
| 2 | Diffusion Policy baseline | _pending_ |
| 3 | + self-value failure detection | _pending_ |
| 3 | + candidate selection | _pending_ |
| 4 | flywheel, 10² trajectories | _pending_ |

Notes on the failures: they cluster on high-return episodes (Seen_5–9, returns 86–199),
i.e. the policy folds substantially but the final state fails the success check — the
known ACT failure mode this project's Stage 3 (self-value prediction) targets.

## Reproduction commands

```bash
# Stage 1 — training (30k steps, ~2.4 h on RTX 5070; batch 16 OOMs on 12 GB, 8 fits at 7.4 GB)
lerobot-train --config_path=configs/train_act.yaml \
  --dataset.video_backend=pyav --batch_size=8

# Stage 1 — evaluation (12 episodes ≈ 12 min; videos land in outputs/eval_videos/{success,failure})
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/top_long_merged \
  --garment_type top_long \
  --num_episodes 1 --save_video --enable_cameras --device cpu
```

Checkpoints: `outputs/train/act_top_long/checkpoints/{005000…030000,last}/pretrained_model`
(zero-padded six-digit names — not `step_N`). Loss curve 3.305 → 1.567 (2k) → 0.517 (5k)
→ 0.252 (12k) → **0.160** (30k).

## Environment pitfalls (hit on this machine, in order)

1. **Blackwell needs cu128.** RTX 5070 is sm_120; stock `torch 2.7.x+cu126` installs fine but
   dies at kernel launch (`no kernel image`). Use `torch==2.7.0+cu128` wheels
   (mirror: `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/`).
2. **Driver = kernel-matched module package.** `nvidia-driver-595-open` alone is not enough;
   also install `linux-modules-nvidia-595-open-$(uname -r)`, then `sudo modprobe nvidia`.
   Driver 580.196.07 (CUDA 13.0) runs the cu128 runtime fine.
3. **Skip `uv sync`.** `[[tool.uv.index]]` hard-pins pypi.org and cannot be overridden from the
   CLI; use plain pip with a mirror (`PIP_INDEX_URL`).
4. **Interrupted pip = zero-byte files.** A killed `pip install` leaves truncated files in
   site-packages that only fail at import. Either use `--no-cache-dir` everywhere or audit
   the venv (scan RECORD files for 0-byte entries) and force-reinstall the damaged packages.
5. **`./isaaclab.sh -i none` needs `export PIP_INDEX_URL=…` first** (it calls pip without a
   mirror), and it upgrades `warp-lang` unpinned — re-pin `warp-lang==1.8.1` afterwards or
   `import warp` collides with omni.warp 1.8.2 (`wp.types.array` ImportError).
6. **Re-pin the pinned stack after every big install**: `numpy==1.26.0` (isaacsim-kernel),
   `pandas==2.2.3` / `pyarrow==17.0.0` (numpy-1.26 compatible), `tokenizers==0.22.2`
   (transformers range), `packaging==23.0` (official override).
7. **Video backend**: default `torchcodec` can't find system ffmpeg libs; use
   `--dataset.video_backend=pyav`.
8. **Eval quirks**: first Isaac Sim launch compiles shaders (5–10 min of silence, don't kill);
   the PhysX particle-cloth warnings are benign; checkpoint dirs are zero-padded (`005000`);
   `--save_video` filenames are always `episode0_*` and overwrite each other.

## Log

- 2026-11-10 — repository created, plan written, compute budget assessed.
- 2026-11-12 — Stage 0 complete: full stack on native Ubuntu 24.04 (Isaac Sim 5.1.0,
  isaaclab 0.53.1, torch 2.7.0+cu128); dataset `top_long_merged` (250 ep / 83,068 frames)
  and Assets downloaded; evaluator smoke-tested with the random policy (12 episodes, 0% success).
- 2026-11-12 — Stage 1 complete: ACT trained 30k steps @ batch 8 on the RTX 5070
  (loss 3.305 → 0.160); evaluated `last` checkpoint over two 12-episode runs —
  83.33% and 75.00% success (run 1: 2/2 on the Unseen variants). Eval videos archived
  in `outputs/eval_videos`.

## License

Code released under Apache 2.0, matching the upstream benchmark. Results and text are my own.
