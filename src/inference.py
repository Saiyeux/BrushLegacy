"""
inference.py  —  PaintTransformer inference (no diffusion, no animation)

Loads 220_net_g.pth, runs on an input image, and outputs one CSV per layer
(layers 3, 4, 5 = large → small brush).

The canvas is NOT rendered between layers (we only extract stroke parameters).
This is a deliberate simplification: each layer sees a white canvas, which
means the model always plans strokes against a blank background.  This is
equivalent to one pass per scale without iterative refinement.

Usage:
    python src/inference.py \
        --image  data/input/painting.png \
        --model  220_net_g.pth \
        --outdir data/strokes/

Output files:
    data/strokes/layer_03_strokes.csv
    data/strokes/layer_04_strokes.csv
    data/strokes/layer_05_strokes.csv

Columns: layer, patch_y, patch_x, stroke_id, d0-d15
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))
from network_diff import Painter, SignWithSigmoidGrad


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Image helpers ──────────────────────────────────────────────────────────

def read_img(path: str, h=None, w=None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if h and w:
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).float() / 255.0


def pad_to(img: torch.Tensor, H: int, W: int) -> torch.Tensor:
    b, c, h, w = img.shape
    ph, pw = (H - h) // 2, (W - w) // 2
    rh, rw = (H - h) % 2, (W - w) % 2
    img = torch.cat([torch.zeros(b, c, ph, w), img,
                     torch.zeros(b, c, ph + rh, w)], dim=2)
    img = torch.cat([torch.zeros(b, c, H, pw), img,
                     torch.zeros(b, c, H, pw + rw)], dim=3)
    return img


# ── CSV writer ─────────────────────────────────────────────────────────────

def write_csv(path: str, layer: int, patch_num: int,
              param_np: np.ndarray, decision_np: np.ndarray,
              p_dim: int) -> int:
    """Write stroke parameters to CSV.  Returns number of strokes written."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = (["layer", "patch_y", "patch_x", "stroke_id"]
              + [f"d{i}" for i in range(p_dim)])
    count = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        idx = 0
        for py in range(patch_num):
            for px in range(patch_num):
                stroke_num = param_np.shape[1]
                for sid in range(stroke_num):
                    if not decision_np[idx, sid]:
                        idx_advance = False
                    else:
                        row = [layer, py, px, sid] + param_np[idx, sid].tolist()
                        w.writerow(row)
                        count += 1
                idx += 1
    return count


# ── Main inference ─────────────────────────────────────────────────────────

def run_inference(image_path: str,
                  model_path: str,
                  out_dir: str,
                  layers: list = None,
                  p_dim: int = 16,
                  stroke_num: int = 8,
                  patch_size: int = 32,
                  score_thresh: float = -0.4) -> dict:
    """Run PaintTransformer inference on one image.

    Args:
        image_path:   input image (any size, will be scaled per layer)
        model_path:   path to 220_net_g.pth
        out_dir:      directory for output CSVs
        layers:       list of layer indices, default [3, 4, 5]
        p_dim:        stroke parameter dimensions (16)
        stroke_num:   strokes per patch (8)
        patch_size:   PaintTransformer patch size (32)
        score_thresh: d12 threshold for decision override (-0.4)

    Returns:
        dict mapping layer → csv_path
    """
    if layers is None:
        layers = [3, 4, 5]

    set_seed(42)
    device = get_device()
    print(f"[inference] device={device}  image={image_path}")

    # Load model
    net_g = Painter(p_dim, stroke_num, 256, 8, 3, 3).to(device)
    state = torch.load(model_path, map_location=device)
    net_g.load_state_dict(state)
    net_g.eval()
    for p in net_g.parameters():
        p.requires_grad_(False)
    print(f"[inference] loaded {model_path}")

    # Load and scale input image
    original_img = read_img(image_path).to(device)
    orig_h, orig_w = original_img.shape[-2:]
    scale = 1024.0 / max(orig_h, orig_w)
    original_img = F.interpolate(original_img,
                                 (int(orig_h * scale), int(orig_w * scale)))
    orig_h, orig_w = original_img.shape[-2:]

    pad_size = patch_size * (2 ** 5)    # 1024: largest possible layer
    img_padded = pad_to(original_img, pad_size, pad_size)

    outputs = {}

    with torch.no_grad():
        for layer in layers:
            layer_size = patch_size * (2 ** layer)
            img = F.interpolate(img_padded, (layer_size, layer_size))

            # White canvas (simplified: no iterative canvas update)
            canvas = torch.ones_like(img)

            img_patches = (F.unfold(img, (patch_size, patch_size),
                                    stride=(patch_size, patch_size))
                           .permute(0, 2, 1).contiguous()
                           .view(-1, 3, patch_size, patch_size))
            canvas_patches = (F.unfold(canvas, (patch_size, patch_size),
                                       stride=(patch_size, patch_size))
                              .permute(0, 2, 1).contiguous()
                              .view(-1, 3, patch_size, patch_size))

            patch_num = (layer_size - patch_size) // patch_size + 1

            stroke_param, stroke_decision = net_g(img_patches, canvas_patches)
            stroke_decision = SignWithSigmoidGrad.apply(stroke_decision)

            # Score-based filtering and sorting
            for i in range(stroke_param.shape[0]):
                p_ = stroke_param[i]
                d_ = stroke_decision[i]
                # Sort by last dimension (score)
                order = torch.sort(p_[:, -1]).indices
                p_ = p_[order]
                d_ = d_[order]
                # Override decision where d12 < threshold (low-quality strokes)
                mask = p_[:, 12] < score_thresh
                d_[mask] = 0
                stroke_param[i]   = p_
                stroke_decision[i] = d_

            param_np    = stroke_param.cpu().numpy()
            decision_np = stroke_decision.squeeze(-1).cpu().numpy().astype(bool)

            name = Path(image_path).stem
            csv_path = str(Path(out_dir) / f"{name}_layer_{layer:02d}_strokes.csv")
            n = write_csv(csv_path, layer, patch_num, param_np, decision_np, p_dim)
            outputs[layer] = csv_path
            print(f"[inference] layer {layer}: {n} strokes → {csv_path}")

    return outputs


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="PaintTransformer inference → per-layer stroke CSV")
    p.add_argument("--image",  required=True, help="Input image path")
    p.add_argument("--model",  default="models/220_net_g.pth",
                   help="PaintTransformer weights (default: models/220_net_g.pth)")
    p.add_argument("--outdir", default="data/strokes",
                   help="Output directory for CSV files (default: data/strokes)")
    p.add_argument("--layers", nargs="+", type=int, default=[3, 4, 5],
                   help="Layers to run (default: 3 4 5)")
    p.add_argument("--score_thresh", type=float, default=-0.4,
                   help="d12 score threshold for discarding strokes (default: -0.4)")
    args = p.parse_args()

    outputs = run_inference(
        image_path   = args.image,
        model_path   = args.model,
        out_dir      = args.outdir,
        layers       = args.layers,
        score_thresh = args.score_thresh,
    )
    print(f"\n[inference] done — {len(outputs)} layers written to {args.outdir}/")


if __name__ == "__main__":
    main()
