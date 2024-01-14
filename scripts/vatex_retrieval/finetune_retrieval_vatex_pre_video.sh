DATATYPE="vatex"
TRAIN_CSV="data/vatex/train.csv"
VAL_CSV="data/vatex/val_retrieval.csv"
DATA_PATH="data/vatex/caption.pickle"
AUDIO_PATH="../data/vatex/audios_16k"
VIDEO_PATH="../data/vatex/video_segments"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/vatex/raw_frames"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_s2_tav_from24/pytorch_model.bin.pretrain.9"
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=26203 \
main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
--epochs=10 --batch_size=128 --n_display=100  \
--model_type audioclip --audio_model audio-clip --init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --cross_model cross-clip \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_video_retrieval/pre_compare  \
--datatype ${DATATYPE} \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func nce  \
--max_audio_length=6  --multi_sentence 10  \
--with_control_token 0.5  --retrieval_finetune loose_seq --train_sim_after_cross \
