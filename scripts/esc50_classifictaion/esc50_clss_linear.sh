DATATYPE="esc50"
TRAIN_CSV="../data/esc50/meta/esc50.csv"
VAL_CSV="../data/esc50/meta/esc50.csv"
AUDIO_PATH="../data/esc50/audio"
OUTPUT_ROOT="ckpts"

#########################audio retrieval ######################################
INIT_MODEL="../models/pre_trained/AudioClip_audioset+howto_with_initial_self_superised_control_token/pytorch_model.bin.pretrain.23"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=26202 \
main_task_audio_classification.py --do_train --num_thread_reader=8   --with_control_token 1 \
--epochs=10 --batch_size=64 --n_display=10  \
--train_csv ${TRAIN_CSV} --pretrained_clip_name ViT-B/32 \
--val_csv ${VAL_CSV} \
--init_model ${INIT_MODEL} \
--audio_path ${AUDIO_PATH}  \
--output_dir ${OUTPUT_ROOT}/ckpt_esc50_audio_classification/test  \
--datatype ${DATATYPE} \
--lr 5e-3 --coef_lr 0  --max_words 32 --batch_size_val 32 \
--max_audio_length=3    \
--freeze_layer_num 13  --audio_rate 16000 \

