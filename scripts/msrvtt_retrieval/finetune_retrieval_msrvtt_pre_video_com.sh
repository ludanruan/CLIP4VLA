DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
AUDIO_PATH="../data/msrvtt/audios_16k"
VIDEO_PATH="../data/msrvtt/videos"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/msrvtt/raw_frames"


########################audio retrieval ######################################


INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25202 \
main_task_video_retrieval.py --do_train  --num_thread_reader=4  --filter_video_id  \
--epochs=5 --batch_size=128 --n_display=100 \
--with_control_token 0.5 --init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32  \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_video_retrieval/for_vis  \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 64 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
--loss_func tav_nce  \
--max_audio_length=16    \