import os
import numpy as np 
import argparse

def normalize(array, axis=-1):
    return array / np.expand_dims(np.linalg.norm(array, axis=axis), axis=axis)

def aggregate(feature):
    feature = normalize(feature, axis=-1)
    return feature.mean(axis=1)

def read_feature(dir):
    audio_features = np.load(f'{dir}/audio_features.npy')
    text_features = np.load(f'{dir}/text_features.npy')
    visual_features = np.load(f'{dir}/visual_features.npy')
    # import pdb;pdb.set_trace()
    audio_features = aggregate(audio_features)
    visual_features = aggregate(visual_features)

    visual_features = normalize(visual_features, axis=-1)
    text_features = normalize(text_features, axis=-1)
    audio_features = normalize(audio_features, axis=-1)
    vids = np.load(f'{dir}/vids.npy')
    return visual_features, text_features, audio_features, vids

def get_sim_matrix(feat1, feat2):
    # import pdb;pdb.set_trace()
    score = np.matmul(feat1, feat2.T)
    return score

def find_similar_audio(dir1, dir2):
    q_text_features, q_visual_features, _, vids_q = read_feature(dir1)
    base_text_features, base_visual_features, base_audio_features, vids_base = read_feature(dir2)
    score_t2t = get_sim_matrix(q_text_features, base_text_features)
    score_t2v = get_sim_matrix(q_text_features, base_visual_features)
    score_t2a = get_sim_matrix(q_visual_features, base_audio_features)

    score_v2t = get_sim_matrix(q_visual_features, base_text_features)
    score_v2v = get_sim_matrix(q_visual_features, base_visual_features)
    score_v2a = get_sim_matrix(q_visual_features, base_audio_features)

    t2t_idx = score_t2t.argmax(axis=-1)
    t2v_idx = score_t2v.argmax(axis=-1)
    t2a_idx = score_t2a.argmax(axis=-1)

    v2t_idx = score_v2t.argmax(axis=-1)
    v2v_idx = score_v2v.argmax(axis=-1)
    v2a_idx = score_v2a.argmax(axis=-1)

    return vids_q, {
        't2t': vids_base[t2t_idx],
        't2v': vids_base[t2v_idx],
        't2a': vids_base[t2a_idx],
        'v2t': vids_base[v2t_idx],
        'v2v': vids_base[v2v_idx],
        'v2a': vids_base[v2a_idx]
    }
    

def main(args):
    print(args)
    vids_q, method2vid = find_similar_audio(args.dir1, args.dir2)
    os.makedirs(args.output, exist_ok=True)
    for method, vids in method2vid.items():
        out_dir = f'{args.output}/{method}'
        os.makedirs(out_dir, exist_ok=True)
        for vid_q, vid in zip(vids_q, vids):
            os.system(f'ln -s {args.audio_root}/{vid}.wav {out_dir}/{vid_q}.wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', help='features of videos without audios', default='ckpts/ckpt_msrvtt_video_retrieval/train_feat_noaudio')
    parser.add_argument('--dir2', help='features of videos with audios', default='ckpts/ckpt_msrvtt_video_retrieval/train_feat_audio')
    parser.add_argument('--output', help='the folder of dir1s audios replaced by dir2 ')
    parser.add_argument('--audio_root', default='/dataset/28d47491/rld/data/vatex/audios_16k')
    args = parser.parse_args()
    main(args)


