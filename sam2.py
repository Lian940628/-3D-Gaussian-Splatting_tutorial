"""
sam2 為 Mask 製作應用，將不需要的地方遮掉，讓 Gaussian Splatting 訓練時不會把不需要的地方也訓練進去。
在執行此程式前，請先完成 Colmap 照片特徵提取串接，並安裝好 sam2 與 conda 環境。
"""

"步驟一 : 建立資料夾"
PROJECT=/home/elaine/drone/0807tree_2
mkdir -p $PROJECT/masks
mkdir -p $PROJECT/sam2_frames
mkdir -p $PROJECT/graphdeco_dataset_masked/images
mkdir -p $PROJECT/graphdeco_dataset_masked/sparse/0

"步驟二 : 準備 SAM2"
conda activate sam2
cd ~/sam2

"步驟三 : 把照片建立成 SAM2 數字序列"
python - <<'PY'
from pathlib import Path
project = Path("/home/elaine/drone/0807LeftGrassTree_2")
src = project / "graphdeco_dataset/images"
dst = project / "sam2_frames"
files = sorted(
list(src.glob("*.jpg")) + list(src.glob("*.JPG")),
key=lambda p: int(p.stem.split("_")[-1])
)
for i, f in enumerate(files):
link = dst / f"{i:05d}.jpg"
if link.exists() or link.is_symlink():
link.unlink()
link.symlink_to(f.resolve())
print("總張數：", len(files))
print(files[0].name, "-> 00000.jpg")
print(files[-1].name, "->", f"{len(files)-1:05d}.jpg")
PY


"步驟四 : 取得第一張照片尺寸"
cd /home/elaine/drone/0807tree_2
nano test_tree2_mask.py


import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor
# ==============
# 專案路徑
# ==============
PROJECT = Path("/home/elaine/drone/0807tree_2")
VIDEO_DIR = str(PROJECT / "sam2_frames")
MASK_DIR = PROJECT / "masks"
CHECKPOINT = "/home/elaine/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT = MASK_DIR / "test_IMG_2708.png"
MASK_DIR.mkdir(parents=True, exist_ok=True)
# ===============
# 第一張照片
#
# IMG_2708.JPG
# Width = 3972
# Height = 2984
#
# BOX 格式：
# [left, top, right, bottom]
# ===============
BOX = np.array(
[500, 520, 3930, 2980],
dtype=np.float32
)
# ==============
# SAM2 Points
#
# label = 1 ：這是樹
# label = 0 ：這是背景
# ==============
POINTS = np.array([
# ==============
# Positive points：樹
# ===============
# 左側向外延伸的枝葉
[800, 1350],
[1050, 1550],
# 左中樹冠
[1350, 1400],
[1500, 1750],
# 上方中央樹冠
[1800, 800],
[2050, 1000],
# 中央主要樹冠
[1850, 1350],
[2100, 1600],
# 右中樹冠
[2600, 1450],
[2900, 1600],
# 右側伸出去的枝葉
[3200, 1050],
[3450, 1400],
# 下方樹冠
[1650, 2050],
[2100, 2050],
[2550, 2000],
# 樹幹
[2200, 2350],
[2200, 2650],
[2200, 2870],
# =================
# Negative points：背景
# =================
# 上方天空
[400, 300],
[1200, 300],
[2200, 250],
[3000, 250],
# 左側建築
[250, 1100],
[350, 1600],
# 右側建築
[3750, 600],
[3800, 1500],
# 左下草地
[400, 2400],
[1000, 2700],
# 右下草地
[3000, 2600],
[3650, 2450],
], dtype=np.float32)
# ====================
# Labels
#
# 前 18 個 = positive
# 後 12 個 = negative
# =====================
LABELS = np.array([
# positive
1, 1,
1, 1,
1, 1,
1, 1,
1, 1,
1, 1,
1, 1, 1,
1, 1, 1,
# negative
0, 0, 0, 0,
0, 0,
0, 0,
0, 0,
0, 0,
], dtype=np.int32)
# ==========================
# 基本檢查
# ==========================
print("============")
print("SAM2 Tree Mask Test")
print("============")
print("Project :", PROJECT)
print("Frames :", VIDEO_DIR)
print("Output :", OUTPUT)
print()
if not Path(VIDEO_DIR).exists():
raise FileNotFoundError(
f"找不到 sam2_frames：{VIDEO_DIR}"
)
if not Path(CHECKPOINT).exists():
raise FileNotFoundError(
f"找不到 SAM2 checkpoint：{CHECKPOINT}"
)
frame_count = len(list(Path(VIDEO_DIR).glob("*.jpg")))
print("SAM2 frames :", frame_count)
if frame_count == 0:
raise RuntimeError("sam2_frames 裡沒有圖片")
# ============================
# GPU
# ============================
if not torch.cuda.is_available():
raise RuntimeError("CUDA 不可用")
device = torch.device("cuda")
print("CUDA :", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0))
print()
# =============================
# 建立 SAM2 Video Predictor
# =============================
print("Loading SAM2...")
predictor = build_sam2_video_predictor(
MODEL_CFG,
CHECKPOINT,
device=device
)
print("SAM2 loaded.")
print()
# ===========================
# 初始化 Video State
# ============================
with torch.inference_mode(), torch.autocast(
"cuda",
dtype=torch.bfloat16
):
print("Initializing video state...")
state = predictor.init_state(
video_path=VIDEO_DIR
)
predictor.reset_state(state)
print("Video state initialized.")
print()
# ===========================
# 第一張影像指定目標樹
# ============================
print("Adding tree prompt...")
frame_idx = 0
obj_id = 1
_, obj_ids, mask_logits = predictor.add_new_points_or_box(
inference_state=state,
frame_idx=frame_idx,
obj_id=obj_id,
points=POINTS,
labels=LABELS,
box=BOX,
)
print("Prompt added.")
# ===========================
# 取得 Mask
# ===========================
mask = mask_logits[0] > 0.0
mask = (
mask
.squeeze()
.cpu()
.numpy()
)
mask = mask.astype(np.uint8) * 255
# ============================
# 儲存 Mask
# ============================
Image.fromarray(mask).save(
OUTPUT,
format="PNG",
compress_level=1
)
# =============================
# 結果資訊
# =============================
white_pixels = np.sum(mask == 255)
black_pixels = np.sum(mask == 0)
total_pixels = mask.size
coverage = white_pixels / total_pixels * 100
print()
print("==========================")
print("Mask 完成")
print("==========================")
print("輸出：", OUTPUT)
print(
"Mask size:",
mask.shape[1],
"x",
mask.shape[0]
)
print("White pixels:", white_pixels)
print("Black pixels:", black_pixels)
print(f"Tree coverage: {coverage:.2f}%")
print()
print("請檢查：")
print("1. 左側伸出的枝葉有沒有完整")
print("2. 中央樹冠有沒有完整")
print("3. 右側枝葉有沒有完整")
print("4. 樹幹有沒有完整")
print("5. 建築、天空、草地是否為黑色")
print()

"步驟五 : 第一張先做 SAM2 Mask 測試"
python -m py_compile /home/elaine/drone/0807tree_2/test_tree2_mask.py

cd ~/sam2
CUDA_VISIBLE_DEVICES=0 python \
/home/elaine/drone/0807tree_2/test_tree2_mask.py

cd /home/elaine/drone/0807tree_2
cp test_tree2_mask.py test_tree2_mask.py.backup_test_ok

"步驟六 : SAM2 全批追蹤"
"第一張 Mask 確認正常後，建立："
nano /home/elaine/drone/0807tree_2/propagate_tree.py

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor
# ==========================
# 路徑設定
# ==========================
PROJECT = Path("/home/elaine/drone/0807tree_2")
FRAME_DIR = PROJECT / "sam2_frames"
MASK_DIR = PROJECT / "masks"
CHECKPOINT = "/home/elaine/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_DIR.mkdir(parents=True, exist_ok=True)
# =========================
# 取得原始圖片名稱
#
# 00000.jpg -> IMG_2708.JPG
# 00001.jpg -> IMG_2709.JPG
# ...
# ========================
original_dir = PROJECT / "graphdeco_dataset/images"
original_files = sorted(
list(original_dir.glob("*.JPG")) +
list(original_dir.glob("*.jpg")),
key=lambda p: int(p.stem.split("_")[-1])
)
print("原始照片數量：", len(original_files))
if len(original_files) == 0:
raise RuntimeError("找不到原始圖片")
# =========================
# 第一張圖片的 Prompt
#
# 這組就是剛才測試成功的設定
# =========================
BOX = np.array(
[500, 520, 3930, 2980],
dtype=np.float32
)
POINTS = np.array([
# -------------------------
# Positive：樹
# -------------------------
[800, 1350],
[1050, 1550],
[1350, 1400],
[1500, 1750],
[1800, 800],
[2050, 1000],
[1850, 1350],
[2100, 1600],
[2600, 1450],
[2900, 1600],
[3200, 1050],
[3450, 1400],
[1650, 2050],
[2100, 2050],
[2550, 2000],
[2200, 2350],
[2200, 2650],
[2200, 2870],
# -------------------------
# Negative：背景
# -------------------------
[400, 300],
[1200, 300],
[2200, 250],
[3000, 250],
[250, 1100],
[350, 1600],
[3750, 600],
[3800, 1500],
[400, 2400],
[1000, 2700],
[3000, 2600],
[3650, 2450],
], dtype=np.float32)
LABELS = np.array([
# Positive
1, 1,
1, 1,
1, 1,
1, 1,
1, 1,
1, 1,
1, 1, 1,
1, 1, 1,
# Negative
0, 0, 0, 0,
0, 0,
0, 0,
0, 0,
0, 0,
], dtype=np.int32)
# =========================================================
# 基本檢查
# =========================================================
print()
print("=======================")
print("SAM2 Tree Propagation")
print("======================")
print("Project :", PROJECT)
print("Frames :", FRAME_DIR)
print("Masks :", MASK_DIR)
if not FRAME_DIR.exists():
raise FileNotFoundError(FRAME_DIR)
if not Path(CHECKPOINT).exists():
raise FileNotFoundError(CHECKPOINT)
frame_files = sorted(FRAME_DIR.glob("*.jpg"))
print("SAM2 frame 數量：", len(frame_files))
if len(frame_files) != len(original_files):
raise RuntimeError(
f"sam2_frames={len(frame_files)}，"
f"原始圖片={len(original_files)}，"
"數量不一致，停止執行。"
)
# ===================================
# GPU
# ===================================
if not torch.cuda.is_available():
raise RuntimeError("CUDA 不可用")
device = torch.device("cuda")
print("CUDA :", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0))
# =============================
# 載入 SAM2
# =============================
print()
print("Loading SAM2...")
predictor = build_sam2_video_predictor(
MODEL_CFG,
CHECKPOINT,
device=device
)
print("SAM2 loaded.")
# ============================
# 初始化 Video State
# ============================
with torch.inference_mode(), torch.autocast(
"cuda",
dtype=torch.bfloat16
):
print()
print("Initializing video state...")
state = predictor.init_state(
video_path=str(FRAME_DIR)
)
predictor.reset_state(state)
print("Video state initialized.")
# ===========================
# 第一幀加入 Prompt
# ===========================
print()
print("Adding prompt to frame 0...")
predictor.add_new_points_or_box(
inference_state=state,
frame_idx=0,
obj_id=1,
points=POINTS,
labels=LABELS,
box=BOX,
)
print("Prompt added.")
# ===========================
# Propagation
# ===========================
print()
print("====================")
print("開始追蹤 206 張影像")
print("====================")
print()
saved = 0
for out_frame_idx, out_obj_ids, out_mask_logits in \
predictor.propagate_in_video(state):
# -------------------------
# 目前只有一棵樹，因此取第一個 object
# ------------------------
mask = out_mask_logits[0] > 0.0
mask = (
mask
.squeeze()
.cpu()
.numpy()
)
mask = mask.astype(np.uint8) * 255
# -----------------------
# 對應回原始檔名
#
# frame 0 -> IMG_2708.png
# frame 1 -> IMG_2709.png
# -------------------------
if out_frame_idx >= len(original_files):
print(
"警告：frame index 超過原始照片數量：",
out_frame_idx
)
continue
original = original_files[out_frame_idx]
output_path = MASK_DIR / (
original.stem + ".png"
)
# ------------------------
# 儲存 binary mask
# ------------------------
Image.fromarray(mask).save(
output_path,
format="PNG",
compress_level=1
)
    saved += 1
    print(
    f"[{saved:03d}/{len(original_files):03d}] "
    f"{output_path.name}"
    )
# =============================
# 完成
# =============================
print()
print("=======================")
print("全部完成")
print("=======================")
print("產生 Mask：", saved, "張")
print("Mask 資料夾：", MASK_DIR)
if saved != len(original_files):
print()
print("WARNING：")
print(
"Mask 數量與原始圖片數量不一致！"
)
else:
print()
print("Mask 數量正確。")
