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

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25602 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4   \
--epochs=10 --batch_size=128 --n_display=1  \
--train_csv ${VAL_CSV} --pretrained_clip_name ViT-B/32 --filter_video_id \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--init_model ${INIT_MODEL} \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/control_token  \
--datatype ${DATATYPE}  \
--lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=6  --multi_sentence 5 \
--with_control_token 1 \
