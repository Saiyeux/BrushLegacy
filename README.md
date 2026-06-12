# BrushLegacy

PaintTransformer 推理 → 笔触处理 → Franka 机器人执行。  
无需 10 GB diffusion 模型，无需逐帧渲染。

---

## 目录结构

```
BrushLegacy/
├── models/
│   ├── 220_net_g.pth          # PaintTransformer 权重 (36 MB)
│   └── u2net.pth              # U2Net 权重 (176 MB, 仅渲染预览用)
├── src/
│   ├── network_diff.py        # Painter 模型定义
│   ├── inference.py           # 推理：image → 16D stroke CSV
│   ├── stroke_convert.py      # 转换：16D CSV → 8D CSV
│   ├── stroke_optimize.py     # 优化：去重叠 + 调色板 + 排序
│   ├── traj_vis.py            # 可视化：NPZ → PNG
│   └── utils.py               # 公共工具
├── data/
│   ├── input/                 # 原始输入图片
│   ├── strokes/               # 中间 CSV 文件
│   ├── trajectories/          # 机器人轨迹 NPZ
│   ├── calibration/           # matrix.npy, xyz_rotated2.npy
│   └── output/                # 最终可视化结果
├── config.yaml
└── requirements.txt
```

---

## 机器分工

| 步骤 | 脚本 | 运行机器 | 依赖 |
|------|------|----------|------|
| 推理 | `inference.py` | MacBook | torch, network_diff.py |
| 16D→8D | `stroke_convert.py` | 任意 | numpy, pandas |
| 优化排序 | `stroke_optimize.py` | 任意 | numpy, pandas |
| 轨迹计算 | `traj_calc.py` *(TODO)* | RT box | pyfranka, matrix.npy |
| 机器人执行 | `robot_exec.py` *(TODO)* | RT box | pyfranka |
| 可视化 | `traj_vis.py` | 任意 | matplotlib |

---

## 环境安装

### MacBook（推理）

```bash
conda create -n brushlegacy python=3.10
conda activate brushlegacy
pip install torch torchvision          # 自动使用 MPS (Apple Silicon)
pip install numpy opencv-python pillow pandas matplotlib scipy pyyaml
```

### RT box（执行，franka_new 环境已有所有依赖）

```bash
conda activate franka_new
# 无需额外安装，已有: numpy, pandas, matplotlib, scipy, opencv, pyfranka
```

---

## 执行方案

所有命令从仓库根目录运行。

### 一键运行（推荐）

```bash
# 跑到笔触预览图（推理 → 转换 → 优化 → 预览），不执行机器人
python run.py --image data/input/painting.png --until preview

# 全流程（包括轨迹计算 + 可视化 + 机器人执行）
python run.py --image data/input/painting.png

# 跳过推理（CSV 已有），只跑后续步骤
python run.py --image data/input/painting.png --skip inference
```

**输出文件（`stem` = 图片文件名去掉扩展名）：**

```
data/strokes/
  {stem}_layer_03_strokes.csv   ← 推理输出（16D）
  {stem}_layer_03_8d.csv        ← 转换输出（8D）
  {stem}_layer_03_sorted.csv    ← 优化排序后
  {stem}_layer_03_removed.csv   ← 被丢弃的笔触
  ...（layer 04, 05 同理）

data/output/
  {stem}_preview.png            ← 笔触预览图（彩色矩形合成）

data/trajectories/
  robot_path_MMDD_HHMMSS.npz   ← 机器人轨迹（traj_calc 生成）
  robot_path_*_overview.png    ← 轨迹可视化图
```

---

### run.py 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--image` | 必填 | 输入图片路径 |
| `--model` | `models/220_net_g.pth` | 模型权重路径 |
| `--layers` | `3 4 5` | 推理层（3=大笔 4=中笔 5=小笔） |
| `--until` | `exec` | 运行到哪步停止：`convert` / `optimize` / `preview` / `vis` / `exec` |
| `--skip` | 无 | 跳过哪些阶段：`inference` / `convert` / `optimize` |

---

### 分步运行（调试用）

```bash
# 单独推理
python src/inference.py --image data/input/painting.png

# 单独转换（一层）
python src/stroke_convert.py --input data/strokes/{stem}_layer_03_strokes.csv \
    --output data/strokes/{stem}_layer_03_8d.csv --layer 3

# 单独优化
python src/stroke_optimize.py --input data/strokes/{stem}_layer_03_8d.csv \
    --output data/strokes/{stem}_layer_03_sorted.csv

# 单独预览
python src/stroke_preview.py \
    --layer3 data/strokes/{stem}_layer_03_sorted.csv \
    --layer4 data/strokes/{stem}_layer_04_sorted.csv \
    --layer5 data/strokes/{stem}_layer_05_sorted.csv \
    --output data/output/{stem}_preview.png

# 轨迹可视化（NPZ 已存在时）
python src/traj_vis.py --npz data/trajectories/robot_path_*.npz
python src/traj_vis.py --npz data/trajectories/robot_path_*.npz --stroke 5
```

---

### RT box 执行（traj_calc + robot_exec 完成后）

```bash
conda activate franka_new
# 需先拷贝 calibration 文件：
#   data/calibration/matrix.npy
#   data/calibration/xyz_rotated2.npy

python run.py --image data/input/painting.png --skip inference convert optimize
# 等效于：traj_calc → traj_vis → robot_exec
```

---

## 调色板（24 色）

```
黑灰白：  Black  Gray  White
标准色：  Green  Red  Purple  Blue  Orange  Yellow  Pink
深色：    DarkGreen  DarkRed  DarkPurple  DarkBlue  DarkOrange  DarkYellow  DarkPink
浅色：    LightGreen  LightRed  LightPurple  LightBlue  LightOrange  LightYellow  LightPink
```

颜色距离超过 180 的笔触会被丢弃。近黑（三通道均 < 0.1）和近白（均 > 250）也丢弃。

---

## TODO

- [x] `src/inference.py` + `src/network_diff.py`
- [x] `src/stroke_convert.py`
- [x] `src/stroke_optimize.py`
- [x] `src/traj_vis.py`
- [ ] `src/traj_calc.py` — 从 `robot/franka_calculate-qinlinset2paperl5.py` 移植
- [ ] `src/robot_exec.py` — 从 `robot/franka22_acldel-qinlinpaperl4spline.py` 移植
- [ ] 拷贝校准文件：`matrix.npy` + `xyz_rotated2.npy` → `data/calibration/`
