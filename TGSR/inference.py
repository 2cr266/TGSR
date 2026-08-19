#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import TGSR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TGSR on one LR image.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_image(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(image * 255.0).astype(np.uint8)).save(path)


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = TGSR(upscale=args.scale).to(device).eval()
    load_checkpoint(model, args.checkpoint)

    lr = load_image(args.input).to(device)
    height, width = lr.shape[-2:]
    pad_h = max(0, 32 - height)
    pad_w = max(0, 32 - width)
    if pad_h or pad_w:
        lr = F.pad(lr, (0, pad_w, 0, pad_h), mode="replicate")

    with torch.inference_mode():
        sr = model(lr)
    sr = sr[..., : height * args.scale, : width * args.scale]
    save_image(sr, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
