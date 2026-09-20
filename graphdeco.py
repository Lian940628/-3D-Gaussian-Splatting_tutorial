"""
GraphDeco Pipeline for 3DGS
在執行此程式前，請先完成 Colmap 照片特徵提取串接，並安裝好 Graphdeco 與 conda 環境。
"""

"步驟一：環境進入"
conda activate graphdeco

"步驟二： Image Undistorter"
mkdir -p graphdeco_dataset
colmap image_undistorter \--image_path images \--input_path /home/elaine/drone/0817softballfield/colmap_manual/sparse/0 \--output_path graphdeco_dataset \--output_type COLMAP

"步驟三： Graphdeco 要求 sparse/0"

mkdir -p graphdeco_data/images
mkdir -p graphdeco_data/sparse/0

tree graphdeco_data

cd /home/elaine/drone/0817softballfield/graphdeco_dataset
mkdir -p sparse/0
mv sparse/cameras.bin \
sparse/images.bin \
sparse/points3D.bin \
sparse/frames.bin \
sparse/rigs.bin \
sparse/0/

"步驟四： 模型訓練"
cd ~/gaussian-splatting

"樹木細節版"
cd ~/gaussian-splatting && python train.py -s /home/elaine/drone/0810droneflytest/graphdeco_dataset -m /home/elaine/drone/0810droneflytest/graphdeco_output --iterations 40000 --densify_until_iter 25000 --position_lr_max_steps 40000 --densify_grad_threshold 0.00015

"室內環繞拍攝一棵樹"
cd ~/gaussian-splatting && CUDA_VISIBLE_DEVICES=1 python tr
ain.py \
 -s /home/elaine/drone/0807tree_3/graphdeco_dataset \
 -m /home/elaine/drone/0807tree_3/graphdeco_output_tree_detail \
 --iterations 50000 \
 --densify_from_iter 500 \
 --densify_until_iter 30000 \
 --densification_interval 100 \
 --densify_grad_threshold 0.00008 \
 --position_lr_init 0.00008 \
 --position_lr_final 0.0000008 \
 --position_lr_max_steps 50000 \
 --scaling_lr 0.003 \
 --opacity_lr 0.03 \
 --feature_lr 0.0025 \
 --lambda_dssim 0.2 \
 --opacity_reset_interval 3000 \
 --save_iterations 10000 20000 30000 40000 50000 \
 --checkpoint_iterations 30000 40000 \
 --antialiasing

"戶外環繞一棵樹 (建議加sam2遮罩把戶外不需要的地方遮掉)"
cd ~/gaussian-splatting
CUDA_VISIBLE_DEVICES=1 python train.py \
-s /home/elaine/drone/0817tree/graphdeco_dataset \
-m /home/elaine/drone/0817tree/graphdeco_output_tree_HQ \
--iterations 50000 \
--densify_until_iter 30000 \
--densify_from_iter 500 \
--densification_interval 100 \
--densify_grad_threshold 0.0001 \
--position_lr_max_steps 50000 \
--opacity_reset_interval 3000 \
--sh_degree 3 \
--lambda_dssim 0.2 \
--save_iterations 10000 20000 30000 40000 50000