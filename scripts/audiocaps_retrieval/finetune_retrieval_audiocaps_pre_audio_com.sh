DATATYPE="audiocaps"
TRAIN_CSV="data/audiocaps/train.csv"
VAL_CSV="data/audiocaps/test.csv"
DATA_PATH="data/audiocaps/captions.csv"
AUDIO_PATH="../data/audiocaps/audios_16k"
OUTPUT_ROOT="ckpts"
VIDEO_PATH="../data/audiocaps/videos"
FEATURES_PATH="../data/audiocaps/raw_frames"

# INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_without_initial/pytorch_model.bin.pretrain.0"


# ########################audio retrieval ######################################
# INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25802 \
# main_task_video_retrieval.py --do_eval  --num_thread_reader=4  --with_control_token 1 \
# --epochs=10 --batch_size=128 --n_display=100 --filter_video_id \
# --model_type audioclip --audio_model audio-clip  --init_model ${INIT_MODEL} \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/pre_compare  \
# --datatype ${DATATYPE}  \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2  \
# --loss_func nce  \
# --max_audio_length=6   --multi_sentence 5 \

# INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_all_self_superised_s2_from24/pytorch_model.bin.pretrain.3"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26303 \
# main_task_video_retrieval.py --do_eval  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/pre_compare  \
# --datatype ${DATATYPE} \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 \
# --loss_func nce  \
# --max_audio_length=10 --multi_sentence 5  \
# --with_control_token 1 --filter_video_id \

# INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_all_self_superised_s2_from24/pytorch_model.bin.pretrain.4"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26303 \
# main_task_video_retrieval.py --do_eval  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/pre_compare  \
# --datatype ${DATATYPE} \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 \
# --loss_func nce  \
# --max_audio_length=10 --multi_sentence 5  \
# --with_control_token 1 --filter_video_id \

# INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_all_self_superised_s2_from24_1e-7/pytorch_model.bin.pretrain.1"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26303 \
# main_task_video_retrieval.py --do_eval  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/pre_compare  \
# --datatype ${DATATYPE} \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 \
# --loss_func nce  \
# --max_audio_length=10 --multi_sentence 5  \
# --with_control_token 1 --filter_video_id \

# INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_all_self_superised_s2_from24_1e-7/pytorch_model.bin.pretrain.2"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26303 \
# main_task_video_retrieval.py --do_eval  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/pre_compare  \
# --datatype ${DATATYPE} \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 \
# --loss_func nce  \
# --max_audio_length=10 --multi_sentence 5  \
# --with_control_token 1 --filter_video_id \

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.6"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26303 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4   \
--epochs=10 --batch_size=128 --n_display=100  \
--init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/pre_compare  \
--datatype ${DATATYPE} \
--lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 \
--loss_func nce  \
--max_audio_length=10 --multi_sentence 5  \
--with_control_token 1 --filter_video_id \