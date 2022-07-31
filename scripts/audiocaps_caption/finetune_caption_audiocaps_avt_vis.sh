DATATYPE="audiocaps"
TRAIN_CSV="data/audiocaps/train.csv"
VAL_CSV="data/audiocaps/test.csv"
DATA_PATH="data/audiocaps/captions.csv"
AUDIO_PATH="../data/audiocaps/audios_16k"
OUTPUT_ROOT="ckpts"
VIDEO_PATH="../data/audiocaps/videos"
FEATURES_PATH="../data/audiocaps/raw_frames"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4  --master_port=25402 \
main_task_video_caption.py \
--do_train  --pretrained_clip_name ViT-B/32 --num_thread_reader=4 \
--epochs=15 --batch_size=64 --do_visualize \
--n_display=100 --filter_video_id --with_control_token 1 \
--init_model ${INIT_MODEL} \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--raw_video_path ${VIDEO_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_caption/for_vis \
--do_lower_case --max_words 32 --max_frames 12 \
--batch_size_val 48 --lr 5e-4 --coef_lr 2e-4  \
--feature_framerate 1  --freeze_layer_num -1  --slice_framepos 2  \
--datatype ${DATATYPE} --stage_two \
--audio_path ${AUDIO_PATH}  --max_audio_length=6   --audio_tokenlen=1 --audio_rate 16000 --audio_channel 2 \

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4  --master_port=25402 \
main_task_video_caption.py \
--do_train  --pretrained_clip_name ViT-B/32 --num_thread_reader=4 \
--epochs=15 --batch_size=64 --do_visualize \
--n_display=100 --filter_video_id --with_control_token 1 \
--init_model ${INIT_MODEL} \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--raw_video_path ${VIDEO_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_caption/for_vis_2 \
--do_lower_case --max_words 48 --max_frames 12 \
--batch_size_val 48 --lr 5e-4 --coef_lr 2e-4  \
--feature_framerate 1  --freeze_layer_num -1  --slice_framepos 2  \
--datatype ${DATATYPE} --stage_two \
--audio_path ${AUDIO_PATH}  --max_audio_length=6   --audio_tokenlen=1 --audio_rate 16000 --audio_channel 2 \
