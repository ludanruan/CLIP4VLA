DATATYPE="audiocaps"
TRAIN_CSV="data/audiocaps/train.csv"
VAL_CSV="data/audiocaps/test.csv"
DATA_PATH="data/audiocaps/captions.csv"
AUDIO_PATH="../data/audiocaps/audios_16k"
OUTPUT_ROOT="ckpts"
VIDEO_PATH="../data/audiocaps/videos"
FEATURES_PATH="../data/audiocaps/raw_frames"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"


# ########################audio retrieval ######################################

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25602 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=8   \
--epochs=15 --batch_size=32 --n_display=100  \
--model_type audioclip --audio_model audio-clip --cross_model cross-clip --with_control_token 1 \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --filter_video_id \
--val_csv ${VAL_CSV} \
--init_model ${INIT_MODEL} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/lr_adjust  \
--datatype ${DATATYPE}  \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=6   --multi_sentence 5 \
--cross_num_hidden_layers 4 --stage_two --train_sim_after_cross \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25602 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=8   \
--epochs=15 --batch_size=32 --n_display=100  \
--model_type audioclip --audio_model audio-clip --cross_model cross-clip --with_control_token 1 \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --filter_video_id \
--val_csv ${VAL_CSV} \
--init_model ${INIT_MODEL} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/lr_adjust  \
--datatype ${DATATYPE}  \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--feature_framerate 1 --coef_lr 1e-3  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=6   --multi_sentence 5 \
--cross_num_hidden_layers 4 --stage_two --train_sim_after_cross \