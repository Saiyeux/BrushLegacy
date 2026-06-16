"""
run.py  —  BrushLegacy 一键执行脚本

从输入图片到机器人执行的完整流程，按阶段可分段运行。

使用方式：

  # 完整流程
  python run.py --image data/input/Tiger.png

  # 只跑到可视化（不执行机器人）
  python run.py --image data/input/Tiger.png --until vis

  # 跳过 gen（已有 sorted CSV），从 preview 开始
  python run.py --image data/input/Tiger.png --skip gen

  # 自定义笔触总数
  python run.py --image data/input/Tiger.png --max_strokes 500

--until 可选值: gen | preview | vis | exec  (默认 exec)
--skip  可选值: gen  (可多选)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
SRC  = ROOT / "src"

STAGES = ["gen", "preview", "vis", "exec"]


def run(cmd: list, step: str) -> None:
    print(f"\n{'='*60}")
    print(f"[{step}] {' '.join(str(c) for c in cmd)}")
    print('='*60)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {step} 失败 (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"[{step}] 完成 ({elapsed:.1f}s)")


def stage_index(name: str) -> int:
    return STAGES.index(name)


def main():
    p = argparse.ArgumentParser(
        description="BrushLegacy pipeline: image → robot execution",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image",       required=True,
                   help="输入图片路径（e.g. data/input/Tiger.png）")
    p.add_argument("--until",       default="exec", choices=STAGES,
                   help="运行到哪个阶段就停止 (default: exec)")
    p.add_argument("--skip",        nargs="*", default=[],
                   choices=["gen"],
                   help="跳过哪些阶段（文件必须已存在）")
    p.add_argument("--max_strokes",  type=int, default=300,
                   help="笔触总数上限 (default: 300)")
    p.add_argument("--dip_interval", type=int, default=10,
                   help="同色每隔多少笔重新蘸墨 (default: 10; 0=禁用)")
    args = p.parse_args()

    image_path  = Path(args.image)
    stem        = image_path.stem
    until_idx   = stage_index(args.until)
    skip_set    = set(args.skip)

    strokes_dir = ROOT / "data" / "strokes"
    traj_dir    = ROOT / "data" / "trajectories"
    output_dir  = ROOT / "data" / "output"

    def sorted_csv(layer):
        return strokes_dir / f"{stem}_layer_{layer:02d}_sorted.csv"

    l3 = sorted_csv(3)
    l4 = sorted_csv(4)
    l5 = sorted_csv(5)

    start_time = datetime.now().strftime("%H:%M:%S")
    print(f"\nBrushLegacy  |  {image_path.name}  |  started {start_time}")
    print(f"until={args.until}  skip={args.skip or 'none'}  max_strokes={args.max_strokes}")

    # ── Step 1: 笔触生成（图像 → sorted 8D CSV） ─────────────────────────────
    if "gen" not in skip_set:
        run([SRC / "stroke_gen.py",
             "--image",       image_path,
             "--outdir",      strokes_dir,
             "--max_strokes", args.max_strokes],
            "gen")
    else:
        print("\n[gen] 已跳过")
    if until_idx == stage_index("gen"):
        print("\n已到达 --until gen，停止。"); return

    # ── Step 2: 笔触预览图 ───────────────────────────────────────────────────
    preview_png = output_dir / f"{stem}_preview.png"
    cmd_preview = [SRC / "stroke_preview.py", "--output", preview_png]
    for layer, path in [(3, l3), (4, l4), (5, l5)]:
        if path.exists():
            cmd_preview += [f"--layer{layer}", path]
    run(cmd_preview, "preview")
    print(f"  → 预览图：{preview_png}")
    if until_idx == stage_index("preview"):
        print("\n已到达 --until preview，停止。"); return

    # ── Step 3: 轨迹计算（sorted CSV → Cobrush Pro NPZ） ────────────────────
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%m%d_%H%M%S")
    latest_npz = traj_dir / f"curves_{stem}_{ts}.npz"

    run([SRC / "traj_calc.py",
         "--layer3",       l3,
         "--layer4",       l4,
         "--layer5",       l5,
         "--output",       latest_npz,
         "--canvas",       512,
         "--max_strokes",  args.max_strokes,
         "--dip_interval", args.dip_interval],
        "traj_calc")

    # ── Step 4: 轨迹可视化 ───────────────────────────────────────────────────
    run([SRC / "traj_vis.py", "--npz", latest_npz], "vis")
    overview = latest_npz.parent / (latest_npz.stem + "_overview.png")
    print(f"  → 轨迹总览：{overview}")
    if until_idx == stage_index("vis"):
        print("\n已到达 --until vis，停止。"); return

    # ── Step 5: 机器人执行 ───────────────────────────────────────────────────
    robot_exec = SRC / "robot_exec.py"
    if not robot_exec.exists():
        print("\n[exec] robot_exec.py 尚未实现，流程停止于可视化阶段。")
        print(f"      NPZ 已就绪：{latest_npz}")
        return

    run([robot_exec, "--npz", latest_npz], "exec")

    total = (datetime.now() - datetime.strptime(start_time, "%H:%M:%S")).seconds
    print(f"\n{'='*60}")
    print(f"  全流程完成  |  {image_path.name}  |  耗时约 {total}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
