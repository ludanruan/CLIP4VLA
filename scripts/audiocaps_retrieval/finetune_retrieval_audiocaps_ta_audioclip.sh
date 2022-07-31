DATATYPE="audiocaps"
TRAIN_CSV="data/audiocaps/train.csv"
VAL_CSV="data/audiocaps/test.csv"
DATA_PATH="data/audiocaps/captions.csv"
AUDIO_PATH="../data/audiocaps/audios_16k"
OUTPUT_ROOT="ckpts"
VIDEO_PATH="../data/audiocaps/videos"
FEATURES_PATH="../data/audiocaps/raw_frames"

#INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised/pytorch_model.bin.pretrain.9"


# ########################audio retrieval ######################################
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=25302 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=10 --batch_size=48 --n_display=10  \
--model_type audioclip --audio_model esresnet   \
--pretrained_clip_name ViT-B/32  \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/audioclip_lr   \
--datatype ${DATATYPE} \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--feature_framerate 1 --coef_lr 1e-3  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=10  --multi_sentence 5 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=25302 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=10 --batch_size=48 --n_display=10  \
--model_type audioclip --audio_model esresnet   \
--pretrained_clip_name ViT-B/32  \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_audio_retrieval/audioclip_lr  \
--datatype ${DATATYPE} \
--lr 1e-5 --max_words 32 --max_frames 12 --batch_size_val 64 \
--feature_framerate 1 --coef_lr 1e-2  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=10  --multi_sentence 5 \