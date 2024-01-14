ROOT_PATH=.
DATA_PATH=${ROOT_PATH}/data/howto100m
SAVE_PATH=${ROOT_PATH}/models
MODEL_PATH=${ROOT_PATH}

INIT_MODEL=""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29700 \
${MODEL_PATH}/main_pretrain.py \
--do_pretrain --num_thread_reader=4 --epochs=10 --with_self_supervised --init_model ${INIT_MODEL} \
--batch_size=4096 --n_pair=1 --n_display=100  \
--model_type audioclip --audio_model audio-clip  --do_lower_case --lr 1e-7 \
--max_words 32 --max_frames 12 --max_audio_length 10  --batch_size_val 512 \
--datatype audioset+howto \
--output_dir ../models/pre_trained/AudioClip_audioset+howto_with_initial_all_self_superised_s2_from24_1e-7 \
--features_path ../data/audioset/raw_frames,../data/Howto100m/raw_frames \
--train_csv data/audioset/AUDIOSET_train.csv,data/howto100m/HOWTO100M_audio_matched.csv \
--data_path data/audioset/caption.pickle,data/howto100m/caption.pickle \
--gradient_accumulation_steps 16 \
--sampled_use_mil  --load_checkpoint \
--audio_path ../data/audioset/audios_16k,../data/Howto100m/audios_cate \
--feature_framerate 1 --coef_lr 1  \
--freeze_layer_num -1  --slice_framepos 2 \
--loss_func tav_nce  \
--with_control_token 2   \

