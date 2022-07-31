DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train_audio.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test_audio.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
FEATURES_PATH="../data/msrvtt/raw_frames"
OUTPUT_ROOT="ckpts"
AUDIO_PATH="../data/msrvtt/audios_16k"
VIDEO_PATH="../data/msrvtt/videos"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=25401 \
main_task_audio_caption.py \
--do_train   --pretrained_clip_name ViT-B/32 --num_thread_reader=4 \
--epochs=10 --batch_size=64  --with_control_token 0.5 \
--n_display=100  --do_visualize --filter_video_id \
--init_model ${INIT_MODEL} \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--raw_video_path ${VIDEO_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_caption/for_vis \
--do_lower_case --max_words 32 --max_frames 12 \
--batch_size_val 64 --lr 5e-4 --coef_lr 2e-4  \
--feature_framerate 1  --freeze_layer_num -1  --slice_framepos 2  \
--datatype ${DATATYPE} --stage_two \
--audio_path ${AUDIO_PATH}  --max_audio_length=16   --audio_tokenlen=1 --audio_rate 16000 --audio_channel 2 \

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=25401 \
main_task_audio_caption.py \
--do_train   --pretrained_clip_name ViT-B/32 --num_thread_reader=4 \
--epochs=10 --batch_size=64  --with_control_token 0.5 \
--n_display=100  --do_visualize --filter_video_id \
--init_model ${INIT_MODEL} \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--raw_video_path ${VIDEO_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_caption/for_vis_2 \
--do_lower_case --max_words 48 --max_frames 12 \
--batch_size_val 64 --lr 5e-4 --coef_lr 2e-4  \
--feature_framerate 1  --freeze_layer_num -1  --slice_framepos 2  \
--datatype ${DATATYPE} --stage_two \
--audio_path ${AUDIO_PATH}  --max_audio_length=16   --audio_tokenlen=1 --audio_rate 16000 --audio_channel 2 \
