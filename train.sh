# # LandmarkGait_Silh_to_Landmark (Silhouette → Landmarks)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
opengait/main.py \
--cfgs ./configs/landmarkgait/LandmarkGait_Silh_to_Landmark.yaml \
--phase train \
--log_to_file

# # LandmarkGait_Landmark_to_Parsing (Landmarks → Parsing)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
opengait/main.py \
--cfgs ./configs/landmarkgait/LandmarkGait_Landmark_to_Parsing.yaml \
--phase train \
--log_to_file

# # LandmarkGait_Recognition
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
opengait/main.py \
--cfgs ./configs/landmarkgait/LandmarkGait_Recognition.yaml \
--phase train \
--log_to_file
