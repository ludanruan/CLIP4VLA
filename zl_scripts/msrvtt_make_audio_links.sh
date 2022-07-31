python find_similar_audio.py \
    --dir1 "/sharefs/multimodel/rld/CLIP4VLA_copy/ckpts/ckpt_msrvtt_video_retrieval/train_feat_noaudio" \
    --dir2 "/sharefs/multimodel/rld/CLIP4VLA_copy/ckpts/ckpt_msrvtt_video_retrieval/train_feat_audio" \
    --output "/sharefs/multimodel/rld/data/msrvtt/audios_complement_16k" \
    --audio_root "/sharefs/multimodel/rld/data/msrvtt/audios_16k"

python find_similar_audio.py \
    --dir1 "/sharefs/multimodel/rld/CLIP4VLA_copy/ckpts/ckpt_msrvtt_video_retrieval/test_feat_noaudio" \
    --dir2 "/sharefs/multimodel/rld/CLIP4VLA_copy/ckpts/ckpt_msrvtt_video_retrieval/train_feat_audio" \
    --output "/sharefs/multimodel/rld/data/msrvtt/audios_complement_16k" \
    --audio_root "/sharefs/multimodel/rld/data/msrvtt/audios_16k"
