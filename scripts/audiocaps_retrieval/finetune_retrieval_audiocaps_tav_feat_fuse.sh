DATATYPE="audiocaps"
TRAIN_CSV="data/audiocaps/train.csv"
VAL_CSV="data/audiocaps/test.csv"
DATA_PATH="data/audiocaps/captions.csv"
AUDIO_PATH="../data/audiocaps/audios_16k"
OUTPUT_ROOT="ckpts"
VIDEO_PATH="../data/audiocaps/videos"
FEATURES_PATH="../data/audiocaps/raw_frames"

INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_s2_tav/pytorch_model.bin.pretrain.4"

########################video retrieval ######################################

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
# main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100 --filter_video_id \
# --model_type audioclip --audio_model audio-clip  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --with_control_token 1 \
# --init_model ${INIT_MODEL} \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
# --datatype ${DATATYPE}  \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2  \
# --loss_func tav_nce  \
# --max_audio_length=6   --multi_sentence 5 \
# --retrieval_finetune query2sim \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
# main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  --filter_video_id \
# --model_type audioclip --audio_model audio-clip  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --with_control_token 1 \
# --init_model ${INIT_MODEL} \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
# --datatype ${DATATYPE}  \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2  \
# --loss_func tav_nce  \
# --max_audio_length=6   --multi_sentence 5 \
# --retrieval_finetune feat2sim \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
--epochs=10 --batch_size=128 --n_display=100  --filter_video_id --with_control_token 1 \
--model_type audioclip --audio_model audio-clip  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--init_model ${INIT_MODEL} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
--datatype ${DATATYPE}  \
--lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func tav_nce  \
--max_audio_length=6   --multi_sentence 5 \
--retrieval_finetune feat_plus \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
# main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  --filter_video_id \
# --model_type audioclip --audio_model audio-clip  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 --with_control_token 1 \
# --init_model ${INIT_MODEL} \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
# --datatype ${DATATYPE}  \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2  \
# --loss_func tav_nce  \
# --max_audio_length=6   --multi_sentence 5 \
# --retrieval_finetune feat2feat \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
# main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  --filter_video_id --with_control_token 1 \
# --model_type audioclip --audio_model audio-clip  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --init_model ${INIT_MODEL} \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
# --datatype ${DATATYPE}  \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2  \
# --loss_func tav_nce  \
# --max_audio_length=6   --multi_sentence 5 \
# --retrieval_finetune query2feat \



# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
# main_task_video_retrieval.py --do_train  --num_thread_reader=4   \
# --epochs=10 --batch_size=128 --n_display=100  --filter_video_id --with_control_token 1 \
# --model_type audioclip --audio_model audio-clip  \
# --train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
# --init_model ${INIT_MODEL} \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --raw_video_path ${VIDEO_PATH} \
# --audio_path ${AUDIO_PATH}  \
# --output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
# --datatype ${DATATYPE}  \
# --lr 2e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
# --feature_framerate 1 --coef_lr 1  \
# --freeze_layer_num -1  --slice_framepos 2  \
# --loss_func tav_nce  \
# --max_audio_length=6   --multi_sentence 5 \
# --retrieval_finetune sim_plus \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
main_task_video_retrieval.py --do_train  --num_thread_reader=8   \
--epochs=15 --batch_size=128 --n_display=100  --filter_video_id --with_control_token 1 \
--model_type audioclip --audio_model audio-clip --cross_model cross-clip \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--init_model ${INIT_MODEL} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
--datatype ${DATATYPE}  \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func tav_nce  \
--max_audio_length=6   --multi_sentence 5 \
--retrieval_finetune  loose_seq --train_sim_after_cross \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
main_task_video_retrieval.py --do_train  --num_thread_reader=8   \
--epochs=15 --batch_size=64 --n_display=100  --filter_video_id --with_control_token 1 \
--model_type audioclip --audio_model audio-clip --cross_model cross-clip \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--init_model ${INIT_MODEL} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
--datatype ${DATATYPE}  \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func tav_nce  \
--max_audio_length=6   --multi_sentence 5 \
--retrieval_finetune  tight_seq --train_sim_after_cross \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25502 \
main_task_video_retrieval.py --do_train  --num_thread_reader=8   \
--epochs=15 --batch_size=64 --n_display=100  --filter_video_id --with_control_token 1 \
--model_type audioclip --audio_model audio-clip --cross_model cross-clip \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--init_model ${INIT_MODEL} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_audiocaps_video_retrieval/feat_fuse \
--datatype ${DATATYPE}  \
--lr 5e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--feature_framerate 1 --coef_lr 2e-4  \
--freeze_layer_num -1  --slice_framepos 2  \
--loss_func tav_nce  \
--max_audio_length=6   --multi_sentence 5 \
--retrieval_finetune  cross_align --train_sim_after_cross \

