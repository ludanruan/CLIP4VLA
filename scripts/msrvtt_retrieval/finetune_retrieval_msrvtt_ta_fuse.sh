DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
AUDIO_PATH="../data/msrvtt/audios_16k"
VIDEO_PATH="./data/msrvtt/videos"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/msrvtt/raw_frames"

#INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"

########################audio retrieval filter_video_id######################################


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25402 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4  --filter_video_id \
# --epochs=5 --batch_size=128 --n_display=100 --with_control_token 0 \
# --model_type audioclip --audio_model audio-clip --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/feat_fuse  \
# --datatype ${DATATYPE} \
# --lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16  \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25402 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  --filter_video_id \
--epochs=10 --batch_size=128 --n_display=1 --with_control_token 0 \
--model_type audioclip --audio_model audio-clip  --init_model ${INIT_MODEL}  \
--train_csv ${VAL_CSV} --pretrained_clip_name ViT-B/32 --cross_model cross-clip \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/feat_fuse  \
--datatype ${DATATYPE} \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
--loss_func nce  \
--max_audio_length=16    \
--retrieval_finetune loose_seq --train_sim_after_cross \


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25402 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4  --filter_video_id \
# --epochs=10 --batch_size=64 --n_display=100 --with_control_token 0 \
# --model_type audioclip --audio_model audio-clip  --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --cross_model cross-clip \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/feat_fuse  \
# --datatype ${DATATYPE} \
# --lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
# --feature_framerate 1 --coef_lr 2e-4  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16    \
# --retrieval_finetune tight_seq --train_sim_after_cross \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25402 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4  --filter_video_id \
# --epochs=10 --batch_size=32 --n_display=100 --with_control_token 0 \
# --model_type audioclip --audio_model audio-clip  --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --cross_model cross-clip \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/feat_fuse  \
# --datatype ${DATATYPE} \
# --lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
# --feature_framerate 1 --coef_lr 2e-4  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16    \
# --retrieval_finetune loss_align --train_sim_after_cross \





