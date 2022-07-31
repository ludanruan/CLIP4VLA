DATATYPE="vatex"
TRAIN_CSV="data/vatex/train.csv"
VAL_CSV="data/vatex/val_retrieval.csv"
DATA_PATH="data/vatex/caption.pickle"
AUDIO_PATH="../data/vatex/audios_16k"
VIDEO_PATH="../data/vatex/video_segments"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/vatex/raw_frames"


INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.5"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.1 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.2 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.3 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.4 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.5 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.6 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.7 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \



CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.8 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 0.9 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25103 \
main_task_audio_retrieval.py --do_train  --num_thread_reader=4  \
--epochs=5 --batch_size=128 --n_display=50  \
--model_type audioclip --audio_model audio-clip --pretrained_clip_name ViT-B/32 \
--train_csv ${TRAIN_CSV}  --init_model ${INIT_MODEL}  --filter_video_id --with_control_token 1 \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--raw_video_path ${VIDEO_PATH} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_audio_retrieval/control_token   \
--datatype ${DATATYPE} \
--lr 1e-7 --max_words 32 --max_frames 12 --batch_size_val 128 \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 --multi_sentence 10 \
--loss_func nce \
--max_audio_length=6    \