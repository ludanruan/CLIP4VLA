DATATYPE="vatex"
TRAIN_CSV="data/vatex/train.csv"
VAL_CSV="data/vatex/val_retrieval.csv"
DATA_PATH="data/vatex/caption.pickle"
AUDIO_PATH="../data/vatex/audios_16k"
VIDEO_PATH="../data/vatex/video_segments"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/vatex/raw_frames"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=26203 \
main_task_visual_retrieval.py --do_eval  --num_thread_reader=4   \
--epochs=5 --batch_size=128 --n_display=100  \
--init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_visual_retrieval/pre_compare  \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 64 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=6  --multi_sentence 10  \
--with_control_token 0.5 --filter_video_id \
