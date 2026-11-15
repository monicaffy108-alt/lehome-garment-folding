"""Stage 4 — retrain the ACT value head with outcome labels.

The Stage 3 progress head regressed `frame_index / (episode_length - 1)`, i.e. it learned
*time*, not *success*, so failures scored as high as successes. Here the label of every
frame is the outcome of its rollout (1 = fold succeeded, 0 = failed) and the loss is BCE.

Only `model.progress_head` is optimised; the ACT backbone stays frozen so the validated
action policy (83.33% success) is untouched.

Usage:
    python scripts/stage4_train_value_head.py \
        --checkpoint outputs/train/act_stage3_v3/checkpoints/030000/pretrained_model \
        --dataset_root Datasets/stage4_rollouts \
        --rollout_root stage4_rollouts \
        --export_checkpoint outputs/train/act_stage3_v4/checkpoints/030000/pretrained_model
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.core import TransitionKey


def resolve_dataset_root(path: Path) -> Path:
    """`--eval_dataset_path` gets a `001/` subdir appended by the recorder; unwrap it."""
    if (path / "meta").exists():
        return path
    candidates = sorted(p for p in path.iterdir() if (p / "meta").exists()) if path.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset under {path}")
    return candidates[-1]


class LabelledRollouts(Dataset):
    """A LeRobot dataset whose frames are labelled with the outcome of their rollout."""

    def __init__(self, root: Path, labels_path: Path, stride: int = 1):
        self.ds = LeRobotDataset(repo_id=f"stage4_{root.name}", root=root, video_backend="pyav")
        success_by_episode = {}
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec["dataset_episode"] is not None:
                        success_by_episode[rec["dataset_episode"]] = rec["success"]
        self.success_by_episode = success_by_episode
        episode_col = self.ds.hf_dataset["episode_index"]
        # Consecutive frames are near-duplicates; stride keeps the head's training set small
        # enough that a full ACT forward per sample stays affordable.
        self.frame_ids = list(range(0, len(episode_col), stride))
        self.episode_of = [int(episode_col[i]) for i in self.frame_ids]

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        item = self.ds[frame_id]
        episode = self.episode_of[idx]
        label = float(self.success_by_episode.get(episode, 0.0))
        return item, torch.tensor(label, dtype=torch.float32)


def to_observation_batch(items, device, action_dim, task):
    """Mirror LeRobotPolicy's preprocessing: numpy/tensor -> normalised batched tensors."""
    obs = {}
    for key, value in items.items():
        if not key.startswith("observation."):
            continue
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        tensor = tensor.float()
        if tensor.ndim == 4 and tensor.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.max() > 1.5:  # uint8 range -> [0, 1]
            tensor = tensor / 255.0
        obs[key] = tensor.to(device)

    batch_size = next(iter(obs.values())).shape[0]
    transition = {
        TransitionKey.OBSERVATION: obs,
        TransitionKey.ACTION: torch.zeros(batch_size, action_dim, device=device),
        TransitionKey.COMPLEMENTARY_DATA: {"task": task},
    }
    return transition


def episode_auc(scores, labels):
    """Probability that a random success scores above a random failure (ties = 0.5)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Stage 3 checkpoint (pretrained_model dir)")
    parser.add_argument("--dataset_root", default="Datasets/stage4_rollouts")
    parser.add_argument("--rollout_root", default="stage4_rollouts")
    parser.add_argument("--export_checkpoint", default="", help="Write a patched checkpoint here")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=3, help="Keep every Nth frame")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--task", default="fold the long sleeve top")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Policy + processors (identical to the evaluation path).
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint, cli_overrides={})
    cfg.pretrained_path = args.checkpoint
    first_garment = resolve_dataset_root(sorted(Path(args.dataset_root).iterdir())[0])
    meta = LeRobotDatasetMetadata(repo_id=f"stage4_{first_garment.name}", root=first_garment)
    policy = make_policy(cfg, ds_meta=meta)
    policy.eval()
    policy.to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    action_dim = policy.config.output_features["action"].shape[0]

    # 2. Freeze everything except the value head.
    for param in policy.parameters():
        param.requires_grad = False
    for param in policy.model.progress_head.parameters():
        param.requires_grad = True
    head_params = [p for p in policy.model.progress_head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=args.lr)

    # 3. Data.
    dataset_root = Path(args.dataset_root)
    rollout_root = Path(args.rollout_root)
    parts = []
    for labels_path in sorted(rollout_root.glob("*.jsonl")):
        garment_dir = dataset_root / labels_path.stem
        if not garment_dir.exists():
            continue
        parts.append(LabelledRollouts(resolve_dataset_root(garment_dir), labels_path,
                                      stride=args.frame_stride))
    dataset = ConcatDataset(parts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    positives = sum(1 for p in parts for v in p.success_by_episode.values() if v)
    negatives = sum(1 for p in parts for v in p.success_by_episode.values() if not v)
    # Failures are the rare class (8 vs 40 rollouts), so weight the two classes equally
    # per *frame* — failures run to max_steps and therefore own a larger share of frames
    # than their episode count suggests.
    frame_labels = [p.success_by_episode.get(ep, 0.0) for p in parts for ep in p.episode_of]
    n_pos = float(sum(frame_labels))
    n_neg = float(len(frame_labels) - n_pos)
    total = float(len(frame_labels))
    w_pos = total / (2.0 * max(n_pos, 1.0))
    w_neg = total / (2.0 * max(n_neg, 1.0))
    print(f"episodes: {positives} success / {negatives} fail")
    print(f"frames: {int(total)} ({int(n_pos)} success / {int(n_neg)} fail) "
          f"-> w_success={w_pos:.2f}, w_fail={w_neg:.2f}")

    # 4. Train the head only.
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        for items, labels in loader:
            transition = to_observation_batch(items, device, action_dim, args.task)
            batch = preprocessor.to_output(preprocessor._forward(transition))
            batch.pop("action", None)  # let ACT sample the latent from the prior, as at inference

            with torch.no_grad():
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)

            # `predict_action_chunk` is wrapped in no_grad, so inline what it does except the
            # gradient shutdown: group the camera keys under OBS_IMAGES, then call the model.
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in policy.config.image_features]
            _, _, progress = policy.model(batch)
            targets = labels.to(device).view(-1, 1)
            sample_weight = torch.where(targets > 0.5, w_pos, w_neg)
            loss = F.binary_cross_entropy(
                progress.clamp(1e-6, 1 - 1e-6), targets, weight=sample_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(labels)
            seen += len(labels)
        print(f"epoch {epoch}: bce={running / max(seen, 1):.4f}")

    # 5. Per-episode predictions — the number that matters is the separation, not the BCE.
    policy.eval()
    scores, labels = [], []
    with torch.no_grad():
        for part in parts:
            for episode, success in sorted(part.success_by_episode.items()):
                idx = [i for i, ep in enumerate(part.episode_of) if ep == episode]
                if not idx:
                    continue
                values = []
                for i in idx[::5]:  # sample every 5th frame
                    items, _ = part[i]
                    items = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in items.items()}
                    transition = to_observation_batch(items, device, action_dim, args.task)
                    batch = preprocessor.to_output(preprocessor._forward(transition))
                    batch.pop("action", None)
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].to(device)
                    policy.predict_action_chunk(batch)
                    values.append(policy._last_progress.item())
                scores.append(float(np.mean(values)))
                labels.append(1.0 if success else 0.0)
                print(f"  {part.ds.root.parent.name} ep{episode}: success={success} "
                      f"value={scores[-1]:.3f} (n={len(values)})")
    print(f"episode-level AUC = {episode_auc(scores, labels):.3f}")

    # 6. Optionally export a checkpoint whose progress head carries the new weights.
    if args.export_checkpoint:
        out = Path(args.export_checkpoint)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(args.checkpoint, out)
        weights_path = out / "model.safetensors"
        state = load_file(str(weights_path))
        for key, value in policy.model.state_dict().items():
            if key.startswith("progress_head."):
                state[f"model.{key}"] = value.detach().cpu()
        save_file(state, str(weights_path))
        print(f"exported -> {out}")


if __name__ == "__main__":
    main()
