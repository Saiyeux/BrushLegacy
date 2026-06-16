"""
traj_sim.py — Frame-by-frame trajectory simulation from action-sequence NPZ.

Reads a brushlegacy_v2 NPZ and renders one PNG per action:
  • PAINT  — stroke appears on canvas; trajectory map updates
  • DIP    — canvas paused; coloured "DIP → Red" banner
  • WASH   — canvas paused; blue "WASH" banner

Output: data/output/{stem}_sim/frame_NNNN.png

Usage:
    python src/traj_sim.py --npz data/trajectories/Tiger_actions.npz
    python src/traj_sim.py --npz data/trajectories/Tiger_actions.npz --step 5
    python src/traj_sim.py --npz data/trajectories/Tiger_actions.npz --image data/input/Tiger.png
"""

import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/vis/

import cv2
import numpy as np

ACTION_PAINT = 0
ACTION_DIP   = 1
ACTION_WASH  = 2

CANVAS_BG = (240, 240, 240)   # light grey trajectory map background


# ── NPZ loader ────────────────────────────────────────────────────────────────

def load_action_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    fmt  = str(data.get("format", b"")) if "format" in data else ""
    if "brushlegacy_v2" not in fmt and "n_actions" not in data:
        raise ValueError(f"{path} is not a brushlegacy_v2 NPZ")
    return {
        "n":            int(data["n_actions"]),
        "W":            int(data["canvas_width"]),
        "H":            int(data["canvas_height"]),
        "types":        data["action_types"],
        "curves":       data["curves"],
        "colors":       data["colors"],
        "widths":       data["widths"],
        "slots":        data["slots"],
        "pal_rgb":      data.get("palette_colors", None),
        "pal_names":    data.get("palette_names",  None),
    }


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_stroke_rect(canvas: np.ndarray, curve, color, width: float) -> None:
    """Render one stroke as a filled rotated rectangle on canvas (RGB in-place)."""
    pts = np.asarray(curve, dtype=np.float32)
    cx  = float(pts[:, 0].mean());  cy = float(pts[:, 1].mean())
    dx  = float(pts[1, 0] - pts[0, 0]);  dy = float(pts[1, 1] - pts[0, 1])
    length    = math.hypot(dx, dy) * 2
    angle_deg = math.degrees(math.atan2(dy, dx))
    bw        = max(4.0, float(width))
    rect      = ((cx, cy), (max(4.0, length), bw), angle_deg)
    box       = cv2.boxPoints(rect).astype(np.int32)
    r, g, b   = int(color[0]), int(color[1]), int(color[2])
    cv2.fillPoly(canvas, [box], (r, g, b))


def _traj_map(all_curves, all_colors, done_up_to: int, W: int, H: int,
              current_curve=None, current_color=None) -> np.ndarray:
    """Rebuild the trajectory map showing all paint strokes up to done_up_to."""
    img = np.full((H, W, 3), CANVAS_BG, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (160, 160, 160), 1)

    # Transit lines (dashed grey)
    prev_end = None
    for i in range(min(done_up_to + 1, len(all_curves))):
        pts = np.asarray(all_curves[i], dtype=np.float32)
        p0  = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        p1  = (int(round(pts[1, 0])), int(round(pts[1, 1])))
        if prev_end is not None:
            dx = p0[0] - prev_end[0];  dy = p0[1] - prev_end[1]
            dist = max(1, int(math.hypot(dx, dy)))
            for t in range(0, dist, 10):
                t0 = t / dist;  t1 = min((t + 6) / dist, 1.0)
                a  = (int(prev_end[0] + dx * t0), int(prev_end[1] + dy * t0))
                b  = (int(prev_end[0] + dx * t1), int(prev_end[1] + dy * t1))
                cv2.line(img, a, b, (200, 200, 200), 1, cv2.LINE_AA)
        prev_end = p1

    # Past strokes
    for i in range(min(done_up_to, len(all_curves))):
        r, g, b = int(all_colors[i][0]), int(all_colors[i][1]), int(all_colors[i][2])
        col_vis  = (max(0, r - 50), max(0, g - 50), max(0, b - 50))
        pts = np.asarray(all_curves[i], dtype=np.float32)
        p0  = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        p1  = (int(round(pts[1, 0])), int(round(pts[1, 1])))
        cv2.line(img, p0, p1, col_vis, 1, cv2.LINE_AA)

    # Current stroke highlight (green)
    if current_curve is not None:
        r, g, b = int(current_color[0]), int(current_color[1]), int(current_color[2])
        pts = np.asarray(current_curve, dtype=np.float32)
        p0  = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        p1  = (int(round(pts[1, 0])), int(round(pts[1, 1])))
        cv2.line(img, p0, p1, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(img, p0, 5, (0, 200, 0), -1, cv2.LINE_AA)

    return img


def _add_banner(paint_panel: np.ndarray, text: str, bg_bgr: tuple,
                text_color: tuple = (255, 255, 255)) -> np.ndarray:
    """Draw a semi-transparent banner across the centre of the panel."""
    H, W = paint_panel.shape[:2]
    out = paint_panel.copy()
    bar_h = 40
    y0    = H // 2 - bar_h // 2
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (W, y0 + bar_h), bg_bgr, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    font_scale = 0.8
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    tx = (W - tw) // 2
    ty = y0 + bar_h // 2 + th // 2
    cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, thick, cv2.LINE_AA)
    return out


def _hud(frame: np.ndarray, action_idx: int, n_total: int,
         atype: int, slot: int, pal_names, pal_rgb,
         paint_done: int) -> None:
    """Top HUD bar with action info."""
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (W, 26), (25, 25, 25), -1)
    aname  = {ACTION_PAINT: "PAINT", ACTION_DIP: "DIP", ACTION_WASH: "WASH"}.get(atype, "?")
    slot_s = ""
    if pal_names is not None and 0 <= slot < len(pal_names):
        slot_s = f"  slot {slot} ({str(pal_names[slot])})"
    txt = (f"Action {action_idx + 1}/{n_total}  [{aname}]{slot_s}"
           f"  |  paint done: {paint_done}")
    cv2.putText(frame, txt, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    # Colour swatch for current slot
    if pal_rgb is not None and 0 <= slot < len(pal_rgb):
        r, g, b = int(pal_rgb[slot, 0]), int(pal_rgb[slot, 1]), int(pal_rgb[slot, 2])
        cv2.rectangle(frame, (W - 36, 4), (W - 6, 22), (r, g, b), -1)
        cv2.rectangle(frame, (W - 36, 4), (W - 6, 22), (160, 160, 160), 1)


# ── Main simulation ───────────────────────────────────────────────────────────

def simulate(npz_path: str, outdir: Path, step: int = 1,
             image_path: str | None = None) -> None:
    info = load_action_npz(npz_path)
    n    = info["n"]
    W    = info["W"];  H = info["H"]
    types    = info["types"]
    curves   = info["curves"]
    colors   = info["colors"]
    widths   = info["widths"]
    slots    = info["slots"]
    pal_rgb  = info["pal_rgb"]
    pal_names = info["pal_names"]

    outdir.mkdir(parents=True, exist_ok=True)

    # Base coat
    if image_path and Path(image_path).exists():
        src = cv2.imread(image_path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA)
        mean_c = src.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    else:
        mean_c = np.array([200, 200, 200], dtype=np.uint8)
    paint_canvas = np.tile(mean_c, (H, W, 1))

    # Palette slot colours for DIP banner
    if pal_rgb is not None:
        slot_rgb = {i: (int(pal_rgb[i, 0]), int(pal_rgb[i, 1]), int(pal_rgb[i, 2]))
                    for i in range(len(pal_rgb))}
    else:
        from palette_cfg import PALETTE_RGB
        slot_rgb = {i: PALETTE_RGB[i] for i in range(len(PALETTE_RGB))}

    # Collect paint-action curves / colors for trajectory map
    paint_curves = []
    paint_colors_list = []

    # Decide which frame indices to save
    must_save = set()
    for i in range(n):
        atype = int(types[i])
        if atype in (ACTION_DIP, ACTION_WASH):
            must_save.add(i)   # always save dip/wash events
    must_save |= {0, n - 1}
    must_save |= set(range(0, n, step))

    paint_done = 0
    current_slot = -1

    print(f"[sim] {n} actions → {len(must_save)} frames  → {outdir}")

    for i in range(n):
        atype = int(types[i])
        slot  = int(slots[i])
        curve = curves[i]
        color = colors[i]
        width = float(widths[i])

        # ── Update state ──────────────────────────────────────────────────────
        current_curve_for_traj = None
        current_color_for_traj = None

        if atype == ACTION_PAINT:
            _draw_stroke_rect(paint_canvas, curve, color, width)
            paint_curves.append(curve)
            paint_colors_list.append(color)
            paint_done += 1
            current_curve_for_traj = curve
            current_color_for_traj = color
            current_slot = slot

        elif atype == ACTION_DIP:
            current_slot = slot

        if i not in must_save:
            continue

        # ── Build frame ───────────────────────────────────────────────────────
        # Left: trajectory map
        traj = _traj_map(paint_curves, paint_colors_list,
                         len(paint_curves),
                         W, H,
                         current_curve_for_traj, current_color_for_traj)

        # Right: paint canvas copy
        paint_panel = paint_canvas.copy()

        if atype == ACTION_DIP:
            r, g, b = slot_rgb.get(slot, (200, 200, 200))
            sname = str(pal_names[slot]) if pal_names is not None else f"slot {slot}"
            paint_panel = _add_banner(paint_panel,
                                      f"DIP  ->  {sname}",
                                      (b, g, r),    # cv2 uses BGR for rectangle fill
                                      (255, 255, 255))
            cv2.rectangle(traj, (2, 2), (W - 3, H - 3), (r, g, b), 3)

        elif atype == ACTION_WASH:
            paint_panel = _add_banner(paint_panel,
                                      "WASH  (brush cleaning)",
                                      (180, 120, 60),   # BGR blue-ish
                                      (255, 255, 255))
            cv2.rectangle(traj, (2, 2), (W - 3, H - 3), (200, 160, 80), 3)

        # Combine side by side
        div   = np.full((H, 4, 3), 180, dtype=np.uint8)
        frame = np.hstack([traj, div, paint_panel])

        # HUD
        _hud(frame, i, n, atype, current_slot, pal_names, pal_rgb, paint_done)

        fname = outdir / f"frame_{i:04d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"[sim] done  →  {outdir}/frame_NNNN.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Frame-by-frame trajectory simulation")
    p.add_argument("--npz",   required=True, help="brushlegacy_v2 action NPZ")
    p.add_argument("--outdir", default=None)
    p.add_argument("--step",  type=int, default=5,
                   help="Save every N-th paint frame (wash/dip always saved, default 5)")
    p.add_argument("--image", default=None,
                   help="Source image for base coat colour")
    args = p.parse_args()

    npz_path = Path(args.npz)
    outdir   = Path(args.outdir) if args.outdir else \
               npz_path.parent.parent / "output" / f"{npz_path.stem}_sim"

    simulate(str(npz_path), outdir, step=args.step, image_path=args.image)


if __name__ == "__main__":
    main()
