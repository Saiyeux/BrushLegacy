"""
calibrate_palette.py — Record physical paint-slot positions for the robot.

For each colour slot the script records TWO poses:
  1. HOVER  — brush safely above the slot, high enough not to touch paint
  2. DIP    — brush touching the paint (the load position)

These match exactly what do_palette() in Cobrush Pro's primitives.py expects:
  joint_go(q_hover)      → fast approach
  joint_go(q_dip)        → slow dip
  sleep(HOLD_SEC)        → load paint
  joint_go(q_hover)      → slow lift

Supports 6 / 12 / 24 physical colour slots.  When fewer than 24 slots are
used, the script automatically computes the nearest-colour mapping from all 24
palette entries to the available physical slots (nearest Euclidean RGB).

Usage:
    # On the RT box with robot connected:
    python src/calibrate_palette.py --colors 12 --ip 192.170.10.200

    # Manual XYZ entry — no robot needed (joint angles will be missing):
    python src/calibrate_palette.py --colors 12 --manual

    # View an existing calibration file:
    python src/calibrate_palette.py --show data/calibration/palette_12.npy

Workflow per slot
-----------------
1. The script prints the target colour (name, RGB, ANSI swatch).
2. You hand-guide the robot (hold wrist guide button) ABOVE the slot.
   Press Enter → records q_hover + pos_hover.
3. You hand-guide the robot DOWN into the paint.
   Press Enter → records q_dip + pos_dip.
4. The script moves to the next slot.

Output: data/calibration/palette_N.npy
  Stored as a dict (np.save with allow_pickle=True).
  Compatible with do_palette() in Cobrush Pro's motion/primitives.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── 24-colour palette — must match COLOR_CENTERS in stroke_gen.py ─────────────
# (palette_index, display_name, RGB)
PALETTE_24: list[tuple[int, str, tuple[int, int, int]]] = [
    ( 0, "Black",       (  0,   0,   0)),
    ( 1, "Gray",        (128, 128, 128)),
    ( 2, "White",       (255, 255, 255)),
    ( 3, "Green",       (  0, 200,   0)),
    ( 4, "Red",         (200,   0,   0)),
    ( 5, "Purple",      (150,   0, 200)),
    ( 6, "Blue",        (  0, 120, 255)),
    ( 7, "Orange",      (255, 165,   0)),
    ( 8, "Yellow",      (255, 255,   0)),
    ( 9, "HotPink",     (255, 100, 180)),
    (10, "DarkGreen",   (  0, 100,   0)),
    (11, "DarkRed",     (100,   0,   0)),
    (12, "DarkPurple",  (100,   0, 150)),
    (13, "RoyalBlue",   (  0,  80, 200)),
    (14, "Brown",       (150,  80,   0)),
    (15, "Olive",       (150, 150,   0)),
    (16, "Crimson",     (200,  50, 120)),
    (17, "LightGreen",  (150, 255, 150)),
    (18, "Salmon",      (255, 100, 100)),
    (19, "LightPurple", (200, 100, 255)),
    (20, "SkyBlue",     (100, 180, 255)),
    (21, "Gold",        (255, 200, 100)),
    (22, "LightYellow", (255, 255, 150)),
    (23, "LightPink",   (255, 150, 200)),
]

# ── Predefined colour subsets (indices into PALETTE_24) ───────────────────────
# Chosen to cover the widest gamut while minimising slot count.
# These are the physical colours you put in the tray.

SUBSET_6 = [0, 2, 4, 3, 6, 7]
# Black, White, Red, Green, Blue, Orange

SUBSET_12 = [0, 1, 2, 4, 3, 5, 6, 7, 8, 11, 14, 18]
# Black, Gray, White, Red, Green, Purple, Blue, Orange,
# Yellow, DarkRed, Brown, Salmon

SUBSET_24 = list(range(24))

SUBSETS = {6: SUBSET_6, 12: SUBSET_12, 24: SUBSET_24}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _swatch(r: int, g: int, b: int) -> str:
    """Three coloured spaces via ANSI 24-bit colour."""
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def _xyz(arr) -> str:
    return f"[{arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f}]"


def compute_mapping(subset: list[int]) -> np.ndarray:
    """Return int32 array of shape (24,) where mapping[i] = slot_index.

    Slot index is the position of the nearest palette colour in `subset`.
    Distance metric: Euclidean RGB.
    """
    slot_rgbs = np.array([PALETTE_24[i][2] for i in subset], dtype=float)
    all_rgbs  = np.array([e[2] for e in PALETTE_24], dtype=float)
    d2 = np.sum((all_rgbs[:, None] - slot_rgbs[None])**2, axis=2)
    return np.argmin(d2, axis=1).astype(np.int32)


# ── Robot helpers ─────────────────────────────────────────────────────────────

def _read_robot_state(api) -> tuple[np.ndarray, list[float]]:
    """Return (pos_xyz, q_joints) from current robot state."""
    st = api.readOnce()
    T  = np.array(st.O_T_EE).reshape(4, 4, order='F')
    return T[:3, 3].copy(), list(st.q)


def _connect_robot(ip: str):
    try:
        from pyfranka.franka_pybind import FrankaApi, RobotMode
    except ImportError:
        print("[ERROR] pyfranka not available on this machine. Use --manual.")
        sys.exit(1)
    api = FrankaApi()
    api.init_config(ip, log_size=1000)
    api.set_default_behavior()
    st = api.readOnce()
    if st.robot_mode.name == "kReflex":
        api.automatic_error_recovery()
    print(f"  Robot connected at {ip}\n")
    return api


# ── Recording ─────────────────────────────────────────────────────────────────

def _record_pose_robot(api, prompt: str) -> tuple[np.ndarray, list[float]]:
    """Show current EE position; wait for Enter; return (pos, q)."""
    pos, q = _read_robot_state(api)
    print(f"     EE now: {_xyz(pos)}")
    input(f"     {prompt}  Press Enter to record… ")
    pos, q = _read_robot_state(api)
    print(f"     Recorded: {_xyz(pos)}")
    return pos, q


def _record_pose_manual(prompt: str) -> tuple[np.ndarray, None]:
    """Prompt user to type XYZ; return (pos, None) — no joint angles."""
    print(f"     {prompt}  Enter x y z (metres, space-separated):")
    while True:
        raw = input("     x y z → ").strip()
        parts = raw.split()
        if len(parts) == 3:
            try:
                return np.array([float(p) for p in parts]), None
            except ValueError:
                pass
        print("     ✗  Need 3 floats, e.g.  0.450 0.123 0.035")


# ── Main calibration loop ─────────────────────────────────────────────────────

def calibrate(n_colors: int, ip: str | None, manual: bool,
              out_path: Path) -> None:
    subset = SUBSETS[n_colors]
    n      = len(subset)

    print(f"\n{'='*64}")
    print(f"  Palette calibration  —  {n} colour slots")
    print(f"  Output → {out_path}")
    print(f"{'='*64}\n")

    # Print slot plan
    print(f"  {'Slot':>4}  {'#':>3}  {'Name':<14}  {'RGB':<22}  Swatch")
    print(f"  {'─'*54}")
    for s, pi in enumerate(subset):
        _, name, rgb = PALETTE_24[pi]
        print(f"  {s:4d}  #{pi:<2d}  {name:<14}  {str(rgb):<22}  {_swatch(*rgb)}")
    print()

    # Connect robot (or not)
    api = None
    if not manual:
        if ip is None:
            print("[ERROR] Provide --ip or use --manual")
            sys.exit(1)
        api = _connect_robot(ip)
        print("  TIP: Hold the wrist guide button to hand-guide the robot.\n"
              "       Release before pressing Enter to record.\n")
    else:
        print("  Manual mode — entering XYZ coordinates only.\n"
              "  Joint angles will not be recorded (palette moves need robot).\n")

    slots_out: list[dict] = []

    for slot_idx, pi in enumerate(subset):
        _, name, rgb = PALETTE_24[pi]
        sw = _swatch(*rgb)

        print(f"\n{'─'*64}")
        print(f"  Slot {slot_idx} / {n-1}   {sw}  {name}  RGB={rgb}")
        print(f"{'─'*64}")

        # ── Step 1: HOVER position ────────────────────────────────────────────
        print("\n  [1/2] HOVER — guide brush above slot (safe height, not touching paint)")
        if api is not None:
            pos_hover, q_hover = _record_pose_robot(api, "Guide to HOVER position.")
        else:
            pos_hover, q_hover = _record_pose_manual("HOVER position above slot:")

        # ── Step 2: DIP position ──────────────────────────────────────────────
        print("\n  [2/2] DIP   — guide brush into the paint")
        if api is not None:
            pos_dip, q_dip = _record_pose_robot(api, "Guide to DIP position (into paint).")
        else:
            pos_dip, q_dip = _record_pose_manual("DIP position (into paint):")

        entry = {
            "name":      name,
            "rgb":       np.array(rgb, dtype=np.uint8),
            "palette_idx": int(pi),
            # Fields used by do_palette() in Cobrush Pro primitives.py:
            "pos_hover": pos_hover,
            "q_hover":   np.array(q_hover) if q_hover is not None else None,
            "pos":       pos_dip,
            "q_dip":     np.array(q_dip)   if q_dip   is not None else None,
        }
        slots_out.append(entry)
        print(f"\n  ✓  Slot {slot_idx} ({name}) saved.")

    # ── Build output dict ─────────────────────────────────────────────────────
    mapping = compute_mapping(subset)

    out_data: dict = {
        "n_slots": n,
        "slots":   slots_out,             # list, index = slot_idx
        "mapping": mapping,               # (24,) int32: mapping[palette_i] = slot_idx
        "subset":  np.array(subset, dtype=np.int32),
    }

    # Backward-compat: keep "black" key for paint.py's pd.get("black")
    for idx, pi in enumerate(subset):
        if pi == 0:  # palette index 0 = Black
            out_data["black"] = slots_out[idx]
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), out_data)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Saved → {out_path}")
    _print_mapping(slots_out, mapping)
    print(f"{'='*64}\n")


# ── Show existing calibration ─────────────────────────────────────────────────

def show(path: Path) -> None:
    if not path.exists():
        print(f"[ERROR] Not found: {path}")
        sys.exit(1)
    cal     = np.load(str(path), allow_pickle=True).item()
    n       = int(cal["n_slots"])
    slots   = cal["slots"]
    mapping = cal["mapping"]

    print(f"\n  Palette calibration: {path.name}  ({n} slots)\n")
    print(f"  {'Slot':>4}  {'Swatch'}  {'Name':<14}  {'RGB':<22}  "
          f"{'Hover XYZ':<36}  Dip XYZ")
    print(f"  {'─'*110}")
    for s, entry in enumerate(slots):
        rgb   = tuple(int(v) for v in entry["rgb"])
        ph    = entry["pos_hover"]
        pd    = entry["pos"]
        ph_s  = _xyz(ph) if ph is not None else "—"
        pd_s  = _xyz(pd) if pd is not None else "—"
        print(f"  {s:4d}   {_swatch(*rgb)}  {entry['name']:<14}  {str(rgb):<22}  "
              f"{ph_s:<36}  {pd_s}")

    print()
    _print_mapping(slots, mapping)


def _print_mapping(slots: list[dict], mapping: np.ndarray) -> None:
    print(f"\n  Nearest-colour mapping (all 24 palette → physical slot)\n")
    print(f"  {'Palette colour':<32}  {'Swatch'}  → {'Slot':>4}  Physical slot")
    print(f"  {'─'*64}")
    for pi, (_, pname, prgb) in enumerate(PALETTE_24):
        slot_idx  = int(mapping[pi])
        slot_name = slots[slot_idx]["name"]
        slot_rgb  = tuple(int(v) for v in slots[slot_idx]["rgb"])
        mark = "←same" if slot_rgb == prgb else ""
        print(f"  #{pi:<2d} {pname:<14} {str(prgb):<16}  {_swatch(*prgb)}  → "
              f"{slot_idx:4d}  {slot_name}  {mark}")


# ── Lookup helper (importable by execution scripts) ───────────────────────────

def load_palette(path: str | Path) -> dict:
    """Load palette.npy and return a callable for colour lookup.

    Returns the raw calibration dict.  Use palette_entry(r, g, b) to get
    the slot dict compatible with do_palette() in Cobrush Pro.

    Example:
        cal = load_palette("data/calibration/palette_12.npy")
        entry = palette_entry(cal, 150, 80, 0)   # Brown stroke
        do_palette(api, entry, "stroke 5")
    """
    return np.load(str(path), allow_pickle=True).item()


def palette_entry(cal: dict, r: int, g: int, b: int) -> dict:
    """Given stroke RGB, return the nearest physical slot entry dict.

    The entry dict has keys: q_hover, pos_hover, q_dip, pos, rgb, name.
    Compatible with do_palette() in Cobrush Pro's motion/primitives.py.
    """
    mapping = cal["mapping"]
    slots   = cal["slots"]
    rgb     = np.array([r, g, b], dtype=float)
    all_rgbs = np.array([e[2] for e in PALETTE_24], dtype=float)
    palette_idx = int(np.argmin(np.sum((all_rgbs - rgb) ** 2, axis=1)))
    slot_idx    = int(mapping[palette_idx])
    return slots[slot_idx]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Record paint-slot positions for the robot palette",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Slot layouts
  6 colours:  Black White Red Green Blue Orange
 12 colours:  + Gray Purple Yellow DarkRed Brown Salmon
 24 colours:  full palette

Examples
  # RT box, 12-colour tray:
  python src/calibrate_palette.py --colors 12 --ip 192.170.10.200

  # MacBook, manual coordinates:
  python src/calibrate_palette.py --colors 6 --manual

  # Review saved calibration:
  python src/calibrate_palette.py --show data/calibration/palette_12.npy
""")
    p.add_argument("--colors", type=int, choices=[6, 12, 24], default=12,
                   help="Number of physical paint slots (default 12)")
    p.add_argument("--ip",     default=None,
                   help="Robot IP (e.g. 192.170.10.200) — required unless --manual")
    p.add_argument("--manual", action="store_true",
                   help="Enter XYZ manually — no robot connection required")
    p.add_argument("--out",    default=None, metavar="PATH",
                   help="Output .npy path  (default: data/calibration/palette_N.npy)")
    p.add_argument("--show",   default=None, metavar="FILE",
                   help="Load and display an existing calibration file, then exit")
    args = p.parse_args()

    if args.show:
        show(Path(args.show))
        return

    out_path = (Path(args.out) if args.out
                else ROOT / "data" / "calibration" / f"palette_{args.colors}.npy")

    calibrate(args.colors, args.ip, args.manual, out_path)


if __name__ == "__main__":
    main()
