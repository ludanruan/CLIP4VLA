DATATYPE="vatex"
TRAIN_CSV="data/vatex/train.csv"
VAL_CSV="data/vatex/no_audio.csv"
DATA_PATH="data/vatex/caption_longest.pickle"
AUDIO_PATH="../data/vatex/audios_16k"
VIDEO_PATH="../data/vatex/videos"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/vatex/raw_frames"


########################audio retrieval ######################################
#############################
#--filter_video_id: dataloader 筛掉没有音频的
#--audio_complement： 给没有视频的音频补上音频
#
INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
#INIT_MODEL="/dataset/28d47491/rld/CLIP4TVA/weights/msvtt_tv_after_finetune.bin.7"
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25110 \
feature_extract.py --do_eval  --num_thread_reader=4 \
--epochs=10 --batch_size=128 --n_display=100 --retrieval_finetune feat_plus \
--model_type audioclip --audio_model audio-clip --with_control_token 0.5 --init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --cross_model cross-clip \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_video_retrieval/no_audio \
--datatype ${DATATYPE} \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
--loss_func tav_nce   \
--max_audio_length=16   \



