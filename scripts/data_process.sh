# #####################frame_extract howto100mt#############################################
# python data_processor.py  --load_video_into_frames --video_dir /dataset/8219588f/howto100m/videos --npy_dir ../data/Howto100m/raw_frames --input_csv ./data/howto100m/HOWTO100M.csv -n 72 >> ../data/Howto100m/extract_frame.log

# #####################frame_extract activitynet#############################################
# python data_processor.py --load_video_into_frames --video_dir ../data/activitynet/videos --npy_dir ../data/activitynet/raw_frames -n 72 >> ../data/activitynet/extract_frame.log

# ######################audio_extract vatex#############################################
python data_processor.py --audio_transfer --video_dir ../data/vatex/videos \
        --audio_dir ../data/vatex/audios_16k --output_json ../data/vatex/audio_transfer.json \
        --input_csv None -n 72 

# #####################frame_extract vatex#############################################
python data_processor.py --load_video_into_frames --video_dir ../data/vatex/videos --npy_dir ../data/vatex/raw_frames -n 72


######################compress_videos audiocaps#############################################
# python data_processor.py --load_video_into_frames --video_dir ../data/audiocaps/videos --npy_dir ../data/audiocaps/raw_frames -n 72 >> ../log/extract_audiocapsframes.log


# ######################audio_extract audiocaps#############################################
# python data_processor.py --audio_transfer --video_dir ../data/audiocaps/videos \
#         --audio_dir ../data/audiocaps/audios_16k --output_json ../data/audiocaps/audio_transfer.json \
#         --input_csv None -n 72 >> ../log/extract_audiocaps_audios.log


######################compress_videos msrvtt#############################################
# python compress_video.py --input_root ../data/msrvtt/videos --output_root ../data/msrvtt/videos_fps3  >> ../log/extract_msrvttframes.log
# python data_processor.py --load_video_into_frames --video_dir ../data/msrvtt/videos --npy_dir ../data/msrvtt/raw_frames -n 72 >> ../log/extract_msrvttframes.log

# ######################audio_extract msrvtt#############################################
# python data_processor.py --audio_transfer --video_dir ../data/msrvtt/videos \
#         --audio_dir ../data/msrvtt/audios_16k --output_json ../data/msrvtt/audio_transfer.json \
#         --input_csv None -n 72 >> ../log/extract_msrvtt_audios.log