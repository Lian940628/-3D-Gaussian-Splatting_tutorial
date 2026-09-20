"""
Spirulae-Splat 為另一種 Gaussian Splatting 的建制方式。
在執行此程式前，請先完成 Colmap 照片特徵提取串接，並安裝好Spirulae-Splat。
"""

"步驟一 : 建立資料夾"
cd /home/elaine/drone/0720grass
mkdir -p spirulae_data/images
mkdir -p spirulae_data/sparse/0

"步驟二 : 複製資料"
cp -r images/* spirulae_data/images/
cp colmap_manual/sparse/0/* spirulae_data/sparse/0/

"步驟三 : 進入專案"
cd ~/spirulae-splat

CUDA_VISIBLE_DEVICES=0 ./build/ssplat-train 3dgs \
 --data /home/elaine/drone/0720grass/spirulae_data \
 --num-iterations 5000 \
 --primitive 3dgs \
 --background-mode noise \
 --background-noise-warmup 1000 \
 --refine-stop-num-iter 500 \
 --max-screen-size 0.15
