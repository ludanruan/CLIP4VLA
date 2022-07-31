DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
AUDIO_PATH="../data/msrvtt/audios_16k"
VIDEO_PATH="./data/msrvtt/videos"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/msrvtt/raw_frames"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_s2_tav/pytorch_model.bin.pretrain.2"


########################audio retrieval ######################################
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25302 \
main_task_visual_retrieval.py --do_train  --num_thread_reader=4   \
--epochs=5 --batch_size=128 --n_display=100  \
--model_type audioclip --audio_model audio-clip --with_control_token 0 --init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_visual_retrieval/lr_adjust  \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
--loss_func nce  \
--max_audio_length=16   

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25302 \
main_task_visual_retrieval.py --do_train  --num_thread_reader=4   \
--epochs=5 --batch_size=128 --n_display=100  \
--model_type audioclip --audio_model audio-clip --with_control_token 0 --init_model ${INIT_MODEL}  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_visual_retrieval/lr_adjust  \
--datatype ${DATATYPE} \
--lr 1e-8 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
--loss_func nce  \
--max_audio_length=16   

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25302 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=5 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --with_bg_token --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/lr_adjust  \
# --datatype ${DATATYPE} \
# --lr 9e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16    \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25302 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=5 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --with_bg_token --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/lr_adjust  \
# --datatype ${DATATYPE} \
# --lr 8e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16    \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25302 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=5 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --with_bg_token --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/lr_adjust  \
# --datatype ${DATATYPE} \
# --lr 7e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16    \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25302 \
# main_task_audio_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=5 --batch_size=128 --n_display=100  \
# --model_type audioclip --audio_model audio-clip --with_bg_token --init_model ${INIT_MODEL}  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_audio_retrieval/lr_adjust  \
# --datatype ${DATATYPE} \
# --lr 6e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2 --expand_msrvtt_sentences \
# --loss_func nce  \
# --max_audio_length=16    \






