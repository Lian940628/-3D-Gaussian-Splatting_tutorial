"""
COLMAP Pipeline for 3DGS
在執行此程式前，請先安裝 COLMAP，並將 COLMAP 的執行檔路徑加入到系統環境變數中。
"""

"步驟一：Feature Extraction"

cd  "開啟自己檔案的位置"

mkdir -p colmap_manual/sparse

CUDA_VISIBLE_DEVICES=0 colmap feature_extractor \
--database_path colmap_manual/database.db \
--image_path images \
--ImageReader.single_camera 1 \
--ImageReader.camera_model OPENCV \
--FeatureExtraction.use_gpu 1 \
--FeatureExtraction.gpu_index 0


"步驟二：Feature Matching"
colmap exhaustive_matcher \
--database_path colmap_manual/database.db \
--FeatureMatching.use_gpu 1
--FeatureExtraction.gpu_index 0

"步驟三：Mapper"
mkdir -p colmap_manual/sparse
colmap mapper \
--database_path colmap_manual/database.db \
--image_path images \
--output_path colmap_manual/sparse

"步驟三：檢查模型"
colmap model_analyzer \--path /home/elaine/drone/0817softballfield/colmap_manual/sparse/0
