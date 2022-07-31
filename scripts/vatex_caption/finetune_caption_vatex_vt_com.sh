DATATYPE="vatex"
TRAIN_CSV="data/vatex/train.csv"
VAL_CSV="data/vatex/test_caption.csv"
DATA_PATH="data/vatex/caption.pickle"
AUDIO_PATH="../data/vatex/audios_16k"
VIDEO_PATH="../data/vatex/video_segments"
OUTPUT_ROOT="ckpts"
FEATURES_PATH="../data/vatex/raw_frames"

#INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.24"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25403 \
main_task_video_caption.py \
--do_train  --model_type=audioclip  --pretrained_clip_name ViT-B/32 --num_thread_reader=4 \
--epochs=10 --batch_size=64  --audio_model audio-clip --cross_model cross-clip   \
--n_display=100 --filter_video_id  --cross_num_hidden_layers 4 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--raw_video_path ${VIDEO_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_vatex_video_caption/layer_adjust \
--do_lower_case --max_words 48 --max_frames 12 \
--batch_size_val 48 --lr 5e-4 --coef_lr 2e-4  \
--feature_framerate 1  --freeze_layer_num -1  --slice_framepos 2  \
--datatype ${DATATYPE} --stage_two \
--audio_path ${AUDIO_PATH}  --max_audio_length=6   --audio_tokenlen=1 --audio_rate 16000 --audio_channel 2 \
