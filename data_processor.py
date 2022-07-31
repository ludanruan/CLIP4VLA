import argparse
import glob, pdb
import json
import cv2
import os
import shutil
import subprocess
import random
import uuid
import shlex
from tqdm import tqdm
import numpy as np
from joblib import delayed
from joblib import Parallel
from moviepy.editor import AudioFileClip, VideoFileClip
from dataloaders.rawvideo_util import RawVideoExtractor
import soundfile as sf
import pandas as pd
import pickle as pkl

AUDIO_SUFFIX=['wav', 'mp3']
VIDEO_SUFFIX=['mp4','webm','mkv']
def audio_transfer_wrapper(audio_in_dir, audio_in,  output_dir):
    """Wrapper for parallel processing purposes."""
    
    #output_filename = construct_video_filename(row, trim_format)
    audio_in_suffix=audio_in.split('.')[-1]
    audio_name = audio_in.split('/')[-1]
    if 'Howto100m' in audio_in_dir:
        output_dir = os.path.join(output_dir, audio_name[0], audio_name[1])
    audio_out = os.path.join(output_dir, audio_in.replace(audio_in_suffix,'wav'))
    audio_in_path = os.path.join(audio_in_dir, audio_in)
    
    if os.path.exists(audio_out):
        status = tuple([audio_out, True, 'Exists'])
        return status
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    try:
        if audio_in_suffix in VIDEO_SUFFIX:
            audio_handle = VideoFileClip(audio_in_path).audio
        elif audio_in_suffix in AUDIO_SUFFIX:
            audio_handle = AudioFileClip(audio_in_path)
        
        audio_handle.write_audiofile(audio_out, fps=16000,  logger=None)
        #use sf to compress audio
        
        wav, rate = sf.read(audio_out)
        
        
        status = tuple([audio_out, "successful"])
        if wav.mean() == 0:
            print(audio_out, ' wav is blank', )
            os.system('rm '+audio_out)
            status = tuple([audio_out, 'failed' , ' wav is blank'])
        

    except Exception as e:
        print(audio_out, ' ', e.args)
        if os.path.exists(audio_out):
            os.system('rm '+audio_out)
        status = tuple([audio_out, 'failed' , e.args])
 
    return status
def build_audio_csv_wrapper(line, dataset='howto100m'):
    video_id, video_feat = line
    video_name = video_feat[:-4]

    video_dir = "/dataset/8219588f/howto100m/videos"
    video_path = os.path.join(video_dir, video_name)

    audio_dir = "/dataset/28d47491/rld/UniVL/data/howto100m/audios_16k"
    audio_path = os.path.join(audio_dir, video_id+'.wav')
    try:
        video_handle = VideoFileClip(video_path)
        audio_handle = AudioFileClip(audio_path)
        
        video_file = os.path.join("/dataset/28d47491/rld/UniVL/data/howto100m/videos_fps3", video_name)
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        cap.release()
        if frame_count> 0 and abs(video_handle.duration - audio_handle.duration)<0.1:
           
            return [line, True]
        
            
    except:
        return [line, False]
    return [line, False]
def modal_available(path):
    modal_available = False
    if os.path.exists(path):
        try:
            if path.split('.')[-1] in VIDEO_SUFFIX:
                cap = cv2.VideoCapture(path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
                cap.release()
                if frame_count> 0:
                    modal_available = True
            elif path.split('.')[-1] in AUDIO_SUFFIX: 
                wav, rate = sf.read(path)
                if wav.mean() != 0:
                   modal_available = True 
        except Exception as e:
            pass
    return modal_available
def build_audioset_csv_wrapper(video_id, video_dir, audio_dir):
    video_available = False
    audio_available = False

    for v_suffix in VIDEO_SUFFIX:
        video_name =  video_id + '.' + v_suffix
        if modal_available(os.path.join(video_dir,video_name)):
            video_available = True
                    
            break
        else:
            video_name_can =  [video_id + '_p.' + v_suffix, video_id + '.' + v_suffix + '.part']
            if modal_available(os.path.join(video_dir,video_name_can[0])):
                video_available = True
                mv_command = ' '.join(['mv', os.path.join(video_dir, video_name_can[0]), os.path.join(video_dir,video_name)])
                os.system(mv_command)
                break
            elif modal_available(os.path.join(video_dir, video_name_can[1])):
                video_available = True
                mv_command = ' '.join(['mv', os.path.join(video_dir, video_name_can[1]), os.path.join(video_dir,video_name)])
                os.system(mv_command)
                break



    audio_name =  video_id + '.wav' 
    if modal_available(os.path.join(audio_dir,audio_name)):
        
        audio_available = True
    else:
        audio_name_can = video_id + '_p.wav'
        if modal_available(os.path.join(audio_dir, audio_name_can)):
            audio_available = True
            mv_command = ' '.join(['mv', os.path.join(audio_dir, audio_name_can), os.path.join(audio_dir,audio_name)])
            os.system(mv_command)

    if audio_available and video_available:
        return [video_id, True, video_name, audio_name]
    else:
        return [video_id, False, video_name, audio_name]



def build_audio_csv(in_csv, out_csv, audio_dir, video_dir=None, sample_num=None, num_jobs=1):
    '''sample sample_num videos to build smaller out_csv, whose corressponding audios are available in audio_dir
        if sample_num is None, then use the whole audios in audio_dir
    '''
    if in_csv.endswith('txt'):
        #clotho, directly transfer txt to csv
        output_csv = pd.DataFrame(columns=["audio_id"])
        with open(in_csv, 'r') as f:
            
            for  line in f.readlines():
                audio_id =line.strip().split('_')[0]
                if not audio_id.isdigit():
                    continue
                data_frame_dict={"audio_id":[audio_id]}
                append_dataframe = pd.DataFrame(data_frame_dict)
                output_csv=output_csv.append(append_dataframe, ignore_index = True)
    elif 'vatex' in in_csv: 
        
        if 'npy' in in_csv:
            video_ids =[item.split('.')[0] for item in np.load(in_csv)] 
        elif 'json' in in_csv:
            video_ids = [item['videoID'] for item in json.load(open(in_csv, 'r'))]

        output_csv = pd.DataFrame(columns=["video_id"])
        for video_id in video_ids:
            data_frame_dict = {"video_id":[video_id]}
            append_dataframe = pd.DataFrame(data_frame_dict)
            output_csv = output_csv.append(append_dataframe, ignore_index = True)      
        # build csv     
    elif in_csv.split('/')[-1] == 'clotho_metadata_development.csv':
        meta_dict = pd.read_csv(in_csv)
        output_csv = pd.DataFrame(columns=["audio_id"])
        for i, row in tqdm(meta_dict.iterrows()):
            audio_id = row["sound_id"]  
            file_name = row["file_name"]
            start_end = row["start_end_samples"]
            if os.path.exists(os.path.join(audio_dir, file_name)) and audio_id.isdigit():
                
                data_frame_dict = {"audio_id":[audio_id]}
                append_dataframe = pd.DataFrame(data_frame_dict)
                output_csv = output_csv.append(append_dataframe, ignore_index = True)
    
    elif 'audiocaps' in in_csv:
        meta_dict = pd.read_csv(in_csv)
        output_csv = pd.DataFrame(columns=["audio_id"])
        for i, row in tqdm(meta_dict.iterrows()):
            audio_id = row["youtube_id"]  
           
            if audio_id  in output_csv["audio_id"].tolist():
                continue
            if os.path.exists(os.path.join(audio_dir, audio_id+'.wav')) :
                
                data_frame_dict = {"audio_id":[audio_id]}
                append_dataframe = pd.DataFrame(data_frame_dict)
                output_csv = output_csv.append(append_dataframe, ignore_index = True)
        
    elif 'howto100m' in in_csv:
        
        csv = pd.read_csv(in_csv)
        csv_keys = ['video_id', 'feature_file']
        output_csv = pd.DataFrame(columns=csv_keys)
        
        #video_ids =[]# save video_id of videos with available audios
        

        # for file in audios:  
        #     video_ids.append(file.split('.')[0])
        # print('read {} video ids with available audios...'.format(len(video_ids)))
       
        assert 'video_id' in csv_keys
        assert 'feature_file' in csv_keys
        status_lst = []
        if num_jobs == 1:
            for line in tqdm(range(len(csv))):
                status_lst.append(build_audio_csv_wrapper([csv['video_id'][line], csv['feature_file'][line]]))
        else:
            status_lst = Parallel(n_jobs=num_jobs)(delayed(build_audio_csv_wrapper)(
            [csv['video_id'][line], csv['feature_file'][line]]) for line in tqdm(range(len(csv))))

        print("Checked if audios and videos of howto100m are matchabale.")
        
        for status_line in tqdm(status_lst):
            line, status = status_line
            data_frame_dict = {}
            if status:
                data_frame_dict = {'video_id':[line[0]], 'feature_file':[line[1]]}
                
                append_dataframe = pd.DataFrame(data_frame_dict)
                output_csv=output_csv.append(append_dataframe, ignore_index = True)
            
                
    elif 'audioset' in in_csv:
    # build video id csv for audioset
        caption_dict = pkl.load(open(in_csv, 'rb'))
        
        video_ids = list(caption_dict.keys())
        status_lst = []
        if num_jobs == 1:
            for line in tqdm(video_ids):
                status_lst.append(build_audioset_csv_wrapper(line, video_dir, audio_dir))
        else:
            status_lst = Parallel(n_jobs=num_jobs)(delayed(build_audioset_csv_wrapper)(
            line, video_dir, audio_dir) for line in tqdm(video_ids))

        print("Checked if audios and videos of audioset are availabel.")


        output_csv = pd.DataFrame(columns=['video_id', 'video_name', 'audio_name'])
        for status_line in tqdm(status_lst):
            video_id, status, video_name, audio_name = status_line
            if status == True:
                data_frame_dict = {'video_id':[video_id], 'video_name':[video_name], 'audio_name':[audio_name]}
                append_dataframe = pd.DataFrame(data_frame_dict)
                output_csv=output_csv.append(append_dataframe, ignore_index = True)
            else:                
                print("{} is not available".format(video_id))       

    elif 'activitynet' in in_csv: 
        video_ids = json.load(open(in_csv, 'r')).keys()

        # status_lst = []
        # if num_jobs == 1:
        #     for line in tqdm(video_ids):
        #         status_lst.append(build_audioset_csv_wrapper(line, video_dir, audio_dir))
        # else:
        #     status_lst = Parallel(n_jobs=num_jobs)(delayed(build_audioset_csv_wrapper)(
        #     line, video_dir, audio_dir) for line in tqdm(video_ids))

        # print("Checked if audios and videos of audioset are availabel.")
        output_csv = pd.DataFrame(columns=['video_id'])
        for video_id in tqdm(video_ids):
            data_frame_dict = {'video_id':[video_id]}
            append_dataframe = pd.DataFrame(data_frame_dict)
            output_csv=output_csv.append(append_dataframe, ignore_index = True)
            
            
    elif in_csv.endswith('csv'):
        csv = pd.read_csv(in_csv)
        csv_keys = csv.keys()
        output_csv = pd.DataFrame(columns=csv_keys)
        
        #video_ids =[]# save video_id of videos with available audios
        

        # for file in audios:  
        #     video_ids.append(file.split('.')[0])
        # print('read {} video ids with available audios...'.format(len(video_ids)))

        assert 'video_id' in csv_keys
       
        for line in tqdm(range(len(csv))):
            video_id = csv["video_id"][line]
            audio_path = os.path.join(audio_dir, video_id+'.wav')
            if os.path.exists(audio_path):
                data_frame_dict = {}
                for key in csv_keys:
                    data_frame_dict[key]=[csv[key][line]]

                append_dataframe = pd.DataFrame(data_frame_dict)
                output_csv=output_csv.append(append_dataframe, ignore_index = True)
            
    print(output_csv)
    output_csv.to_csv(out_csv)

def build_list_from_csv(csv_name, audio_dir, cp_dir="/dataset/28d47491/rld/data/audiocaps/videos"):
    key_list={'audiocaps':"youtube_id", 'howto100m':'video_id'}
    csv = pd.read_csv(csv_name)
    audio_list = []
    audio_key = None
    for key in key_list.keys():
        if key in csv_name:
            audio_key = key_list[key]
            break
    assert audio_key is not None
    if key == 'audiocaps':
        for line in tqdm(range(len(csv))):
            
            output_path = os.path.join(cp_dir,csv[audio_key][line]+'.mp4')
            if os.path.exists(output_path):
                audio_list.append(output_path)
                continue
            video_path = os.path.join(audio_dir,csv[audio_key][line]+'.mp4')
            cp_command = ' '.join(['cp', video_path, output_path])
            sub_out = subprocess.run(cp_command, shell=True, stdout=subprocess.PIPE)
            if sub_out.returncode != 0:
                video_path = os.path.join(audio_dir,csv[audio_key][line]+'_p.mp4')
                cp_command = ' '.join(['cp', video_path, output_path])
                sub_out = subprocess.run(cp_command, shell=True, stdout=subprocess.PIPE)

            if sub_out.returncode != 0:
                print("{} is not exists in {}".format(csv[audio_key][line], audio_dir))
            else:
                audio_list.append(output_path)
    
    elif key == 'howto100m':
        
        #从csv 中把video_id全部扒拉下来
        for line in tqdm(range(len(csv))):
            audio_list.append(csv[audio_key][line]+'.wav')
            

    return audio_list

def audio_transfer(input_dir, output_dir, input_csv=None, output_json=None, num_jobs=20):
    '''transfer video to audio from input dir to output_dir with num_jobs threads,
       following the file tree orgnization of input_dir '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("prepare audio list...")    
    if os.path.exists(input_csv):
        audio_list = build_list_from_csv(input_csv, input_dir)
    else:
        audio_list = os.listdir(input_dir)
    # audio_list = ["aA5DA5WhszE.wav", "baCBr20b26g.wav"]
    # for root, dirs, files in os.walk(input_dir):
    #     for single_file in files:  
    #         audio_list.append(os.path.join(root.replace(input_dir, '.'), single_file))    
   
    # Transfer all videos to audios.
    if num_jobs == 1:
        status_lst = []
        for row in tqdm(audio_list):
            status_lst.append(audio_transfer_wrapper(input_dir, row, output_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(audio_transfer_wrapper)(
           input_dir, row, output_dir) for row in tqdm(audio_list))

    # Save transfer report.
    with open(output_json, 'w') as fobj:
        fobj.write(json.dumps(status_lst))

def build_audio_statistics(audio_dir,  output_json,  sample_num=None):
    '''Collect basic information about audio in audio_dir, 
    including the average length and two-channel variance'''
    
    audios = os.listdir(audio_dir)
    if sample_num is not None:
        audios = audios[:sample_num]
      

    statistics= {
        'audio_dir': audio_dir,
        'ave_duration': 0,
        'channel_statics':{
            'left':{
                'mean': 0,
                'var':0
            },
            'right': {
                'mean':0,
                'var':0
            },
            'l1_distance':0,
            'l2_distance':0
        }
    }
    for audio in tqdm(audios):
       
        raw_wav, rate = sf.read(os.path.join(audio_dir, audio))
        duration = raw_wav.shape[0]/rate * 1.0
        left_mean = np.mean(raw_wav[:,0:1])
        left_var = np.var(raw_wav[:,0:1])
        right_mean = np.mean(raw_wav[:,1:2])
        right_var = np.var(raw_wav[:,1:2])
        l1_distance = np.sum(abs(raw_wav[:,0:1]-raw_wav[:,1:2])) /duration
        l2_distance = np.sqrt(np.sum((raw_wav[:,0:1]- raw_wav[:,1:2])**2)) /duration

        statistics['ave_duration'] += duration
        statistics['channel_statics']['left']['mean'] += left_mean
        statistics['channel_statics']['left']['var'] += left_var
        statistics['channel_statics']['right']['mean'] += right_mean
        statistics['channel_statics']['right']['var'] += right_var
        statistics['channel_statics']['l1_distance'] += l1_distance
        statistics['channel_statics']['l2_distance'] += l2_distance

    statistics['ave_duration'] /= len(audios)
    statistics['channel_statics']['left']['mean'] /= len(audios)
    statistics['channel_statics']['left']['var'] /= len(audios)
    statistics['channel_statics']['right']['mean'] /= len(audios)
    statistics['channel_statics']['right']['var'] /= len(audios)
    statistics['channel_statics']['l1_distance'] /= len(audios)
    statistics['channel_statics']['l2_distance'] /= len(audios)

    with open(output_json, 'w') as fobj:
        fobj.write(json.dumps(statistics, indent=4))
    return

def build_annotation_statistics(audio_dir, annotation_path, output_json):
    datasets = ['howto100m', 'youcookii', 'msrvtt']
    dataset = None
    for item in datasets:
        if  audio_dir.find(item) > 0:
            dataset = item
            print('build annotation dict for dataset {}'.format(dataset))
            break
    
    annotation_statistics = {
        'train':{
            'num':0,
            'mean_duration':0,
            'no_audio':0
        },
        'dev':{
            'num':0,
            'mean_duration':0,
            'no_audio':0
        },
        'test':{
            'num':0,
            'mean_duration':0,
            'no_audio':0
        },
        'all':{
            'num':0,
            'mean_duration':0,
            'no_audio':0
        }

    }

    if annotation_path.endswith('json'):
        # only for msrvtt
        annotations = json.load(open(annotation_path, 'r'))['videos']        
    elif annotation_path.endswith('pkl') or annotation_path.endswith('pickle'):
        annotations = pkl.load(open(annotation_path, 'rb')) 
        
       

    def update_annotation_statistics_with_single_video(segment_anno, video_type):
        assert video_type in annotation_statistics.keys()
        annotation_statistics[video_type]['num'] += 1
        annotation_statistics[video_type]['mean_duration'] += segment_anno['end'] - segment_anno['start']
        video_id = segment_anno['video_id']
        audio_path = os.path.join(audio_dir, video_id + '.wav')
        if not os.path.exists(audio_path):
            annotation_statistics[video_type]['no_audio'] += 1
        return

    if dataset == 'msrvtt':
        for annotation in tqdm(annotations):
            segment_anno = {'video_id':annotation['video_id'], 'start': annotation['start time'], 'end': annotation['end time'], 'split':annotation['split'] }
            update_annotation_statistics_with_single_video(segment_anno, segment_anno['split'])
            update_annotation_statistics_with_single_video(segment_anno, 'all')
    elif dataset == 'youcookii':
        for vid, annotation in tqdm(annotations.items()):
           for s,e in zip(annotation['start'], annotation['end']):
               segment_anno = {'video_id':vid, 'start': s, 'end': e}
               update_annotation_statistics_with_single_video(segment_anno, 'all')
    elif dataset == 'howto100m':
        for vid, annotation in tqdm(annotations.items()):
           for s,e in zip(annotation['start'], annotation['end']):
               segment_anno = {'video_id':vid, 'start': s, 'end': e}
               update_annotation_statistics_with_single_video(segment_anno, 'all')
        
            
        
    for key, item in annotation_statistics.items():
        annotation_statistics[key]['mean_duration'] /= (annotation_statistics[key]['num']+0.0001)

    print(annotation_statistics)  
    with open(output_json, 'w') as fobj:
        fobj.write(json.dumps(annotation_statistics, indent=4))
    return

def csvs_combine(in_csvs, out_csv):
    assert len(in_csvs) > 1
    all_csv = [pd.read_csv(in_csvs[0])]
    csv_keys = all_csv[0].keys()
    for csv_name in in_csvs[1:]:
        csv = pd.read_csv(csv_name)
        
        assert set(csv.keys()) == set(csv_keys)
        all_csv.append(csv)
    all_csv = pd.concat(all_csv, axis=0, ignore_index=True)
    all_csv.to_csv(out_csv,index=False)

def np64tonp16_wrapper(npy_dir, npy_name, new_npy_dir):
    npy_in = os.path.join(npy_dir, npy_name)
    npy_out = os.path.join(new_npy_dir, npy_name)
    if os.path.exists(npy_out) or not os.path.exists(npy_in):
        return
    npy= np.load(npy_in).astype(np.float16)
    np.save(npy_out,npy)
    rm_command = ' '.join(['rm', '-r', npy_in])
    subprocess.run(rm_command, shell=True)
    
    
def np64tonp16(npy_dir, new_npy_dir, input_csv,num_jobs=1):
    if input_csv is None:
        npy_list = os.listdir(npy_dir)
    else:
        csv =  pd.read_csv(input_csv)
        assert "video_id" in csv.keys()
        
        npy_list = []
        for line in tqdm(range(len(csv))):
            npy_list.append(csv["video_id"][line]+'.npy')
            
    print("{} npy waiting to be transfered from {} to {}".format(len(npy_list), npy_dir, new_npy_dir))

    if num_jobs == 1:
        
        for npy_name in tqdm(npy_list):
            np64tonp16_wrapper(npy_dir, npy_name, new_npy_dir)
    else:
         Parallel(n_jobs=num_jobs)(delayed(np64tonp16_wrapper)(
           npy_dir, npy_name, new_npy_dir) for npy_name in tqdm(npy_list))
    mv_command = ' '.join(['mv', new_npy_dir, npy_dir])
    subprocess.run(mv_command, shell=True)
    
    
def generate_video_segments(video_dir, new_video_dir, data_path):
    if data_path.endswith('pickle') or ata_path.endswith('pkl'):
        data_dict = pkl.load(open(data_path, 'rb')) 
    video_dict = dict([[vid[:11], vid] for vid in os.listdir(video_dir)])
    
    for video_id, data_dict in tqdm(data_dict.items()):
        for s, e in zip(data_dict['start'], data_dict['end']):
            
            s = str(s)
            e = str(e)
            output_path = os.path.join(new_video_dir,'_'.join([video_id, s, e])+'.mp4')
            if os.path.exists(output_path):
                continue
            #ffmpeg -i source-file.foo -ss 0 -t 600 -c  first-10-min.m4v
            try:
                ffmpeg_command = ' '.join(['ffmpeg',  '-i', os.path.join(video_dir,video_dict[video_id]), \
                            '-ss', s, '-to', e, '-c', 'copy', output_path])

                subprocess.run(ffmpeg_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
    
def csv2lst(input_csv, output_csv, npy_dir):
    csvs = input_csv.split(',')
    suffixes = {'howto100m':['wav'], 'youcookii':['mp4', 'mkv'], 'msrvtt':['mp4'], 'audiocaps':['npy'], 'vatex':['mp4']}
    id_keys = {'howto100m':'video_id', 'youcookii': 'video_id', 'msrvtt':'video_id', 'audiocaps':'youtube_id', 'vatex':'videoID'}
    datatype = None
    id_key = None
    for suffix_key in suffixes.keys():
        if suffix_key in input_csv:
            datatype = suffix_key
            id_key = id_keys[suffix_key]
            break
    assert datatype is not None and id_key is not None, "!!check your input csv pathes..."

    with open(output_csv, 'w') as fout:
        for csv in csvs:
            csv =  pd.read_csv(csv)
            assert id_key in csv.keys()
                    
            
            for line in tqdm(range(len(csv))):
                video_path=None
                for suf in suffixes[datatype]:
                    video_path = os.path.join(npy_dir,csv[id_key][line]+'.'+suf)
                    if os.path.exists(video_path):
                        fout.write(video_path+'\n')
                        break
                if video_path is None:
                    print("there is no raw data for video {}".format(csv[id_key][line]))
                
def csvs_split(input_csv, output_csvs, rates=[0.99,0.01]):
    '''
    split csv into output_csvs, lines in output_csvs are not overlapped with each other.
    '''
    assert len(output_csvs) == len(rates)
    in_csv = pd.read_csv(input_csv)
    csv_keys = in_csv.keys()
    line_ids = list(range(len(in_csv)))
    random.shuffle(line_ids) 
    sid = 0
    
    for output_csv, rate in zip(output_csvs, rates):
        group_num = round(len(in_csv) * rate)
        assert sid+group_num <= len(line_ids)
        output_pd = in_csv.loc[sid:sid+group_num]
        output_pd.to_csv(output_csv, index = False)
        sid = sid+group_num   
        print(output_pd)
        print("save {}".format(output_csv))
       
        #write lines_ids[sid, group_num+sid] to corressponding output_csv


def build_pkl_subset(in_csv_path,in_pkl_path,out_pkl_path):
    in_pkl = pkl.load(open(in_pkl_path, 'rb'))
    in_csv = pd.read_csv(in_csv_path)
    video_ids = []
    for i, row in tqdm(in_csv.iterrows()):
        video_ids.append(row["video_id"])
    new_dict={}
    for video_id in video_ids:
        new_dict[video_id]=in_pkl[video_id]
    
    with open(out_pkl_path, 'wb') as fo:     # 将数据写入pkl文件
        pkl.dump(new_dict, fo)
    print('save sub dict on {}'.format(out_pkl_path))  

def build_annotation(input_csvs,  output_pkl,input_pkl=None):
    '''
    This function is only for audioset/activitynet/vatex, build output_pkl ini formation of:
    {'video_id':{'text':['label1','label2']}}
    '''
    if 'audioset' in output_pkl:
        input_csv = input_csvs[0]
        assert 'audioset' in input_csv and 'audioset' in input_pkl and 'audioset' in output_pkl 
    #build label dict: {'/m/04zc0':{'text': 'mp3', 'child_ids': []}}
        label_list = json.load(open(input_pkl, 'r',encoding='utf-8'))
        label_dict = {}
        for label_item in label_list:
            label_dict[label_item['id']]={'text':label_item['name'], 'child_ids': label_item['child_ids']}
        
        output_dict={}
        with open(input_csv, 'r') as in_csv:
            for line in tqdm(in_csv.readlines()):
                
                video_id, start_time, end_time, video_labels = line.strip().split(', ')
                video_labels = video_labels.replace('"','').split(',')
                video_label_names = ','.join([label_dict[item]['text'] for item in video_labels])
                output_dict[video_id]={'starts':[int(start_time.split('.')[0])], 'ends':[int(end_time.split('.')[0])], 'text':[video_label_names]}
    elif 'activitynet' in output_pkl:
        all_dict={}
        for input_csv in input_csvs:
            input_dict = json.load(open(input_csv, 'r'))
            all_dict = dict(all_dict,**input_dict) 
        output_dict = {}
       
        for key,value in all_dict.items():
            timestamps = value['timestamps']
            if 'concepts' in value.keys():
                scripts = [','.join(script) for script in value['concepts']]
            else: scripts = ['' for i in range(len(value['timestamps']))]
            captions = value['sentences']
            start = [i[0] for i in timestamps]
            end = [i[1] for i in timestamps]

            output_dict[key]={'start':start, 'end': end, 'captions':captions, 'scripts': scripts}
    elif 'vatex' in output_pkl:
        
        all_list=[]
        for input_csv in input_csvs:
            input_list = json.load(open(input_csv, 'r'))
            all_list.extend(input_list) 
        output_dict = {}
        for item in all_list:
            key = item['videoID']
            captions = item['enCap']
            output_dict[key]={'captions':captions}

    
    with open(output_pkl, 'wb') as fout:     # 将数据写入pkl文件
        pkl.dump(output_dict, fout)
    
    print('output caption annotations to {}'.format(output_pkl))

    return

def load_video_into_frames_wrapper(video_path, npy_dir, feature_framerate = 1, image_resolution=224):
    
    rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
    
    video_name = video_path.split('/')[-1]
    video_id = video_name.split('.')[0]
    if 'howto100m' in video_path or 'audioset' in video_path:
        frame_dir = os.path.join(npy_dir, video_id[0], video_id[1], video_id)
    else:
        frame_dir = os.path.join(npy_dir,  video_id)
    
    if os.path.exists(frame_dir):
        return [video_name, True]

    os.makedirs(frame_dir)

    try:
        raw_video_slices = rawVideoExtractor.extract_frames(video_path, sample_fp = 1)
           
    except:
        print("read {} failed".format(video_path))
        raw_video_slices = None
        return [video_name, False]

   
    if raw_video_slices is not None:
        
        
        for idx, frame in enumerate(raw_video_slices):
            frame_path = os.path.join(frame_dir, str(idx)+'.jpg')
            cv2.imwrite(frame_path, frame)
       
    return [video_name, True]

def load_video_into_frames(video_dir, frame_dir, video_id_csv, num_jobs=1):
    
     
    status_lst = []

    if video_id_csv is not None and os.path.exists(video_id_csv):
        video_csv = pd.read_csv(video_id_csv)
        if 'audioset' in video_id_csv:
            video_names = [video_name for video_name in video_csv['video_name']]
        elif 'howto100m' in video_id_csv:#24+16w+20+5:-20w
            video_names = [video_name[:-4] for video_name in video_csv['feature_file']]
    else:
        video_names = os.listdir(video_dir)
    if num_jobs == 1:
        for video_name in tqdm(video_names):
            video_path= os.path.join(video_dir, video_name)
            status_lst.append(load_video_into_frames_wrapper(video_path, frame_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(load_video_into_frames_wrapper)(
            os.path.join(video_dir, video_name), frame_dir) for video_name in tqdm(video_names))       

    print("extracting frames from {} to {} has finished".format(video_dir, frame_dir))       
    return
            
def audio_csv_minus(original_csv, minus_csv, output_csv):
    '''
    delete the same video_id in minus_csv from original_csv and output as output_csv
    '''
    original_csv = pd.read_csv(original_csv)
    minus_csv = pd.read_csv(minus_csv)
    for line in minus_csv.iterrows():
        delete_index = origianl_csv[origianl_csv['video_id']==line['video_id']].index
        if len(delete_index) > 0:
            # 从 original_csv 中 删除delete_index
            original_csv = original_csv.drop(delete_index)
    original_csv.to_csv(output_csv)    
    
    return
def mv_wrapper(para):
    from_path, to_path = para
    video_id = from_path.split('/')[-1].split('.')[0]
        
    if os.path.exists(to_path) or os.path.exists(from_path) is False:
        return [ 'done',video_id]
    root_dir = os.path.dirname(to_path)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok = True)
    cp_command = ' '.join(['mv',from_path, to_path])
    os.system(cp_command)
   
    return [ 'successful', video_id]
def build_catalog(input_dir, output_dir, input_csv, num_jobs=1):
    video_ids = pd.read_csv(input_csv)['video_id']
    from_list = []
    to_list = []
    if 'audios' in  input_dir:
        post = '.wav'
    else:
        raise NotImplementedError
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for video_id in tqdm(video_ids):
        from_path = os.path.join(input_dir, video_id+'.wav')
        to_path = os.path.join(output_dir,video_id[0], video_id[1],video_id+'.wav')
        from_list.append(from_path)
        to_list.append(to_path)
    status_lst=[]
    if num_jobs == 1:
        for from_path, to_path in tqdm(zip(from_list, to_list)):
            status_lst.append(mv_wrapper([from_path, to_path]))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(mv_wrapper)(
            [from_path, to_path]) for from_path, to_path in tqdm(zip(from_list, to_list)))
    return
    
if __name__ == '__main__':
    description = 'Processors for UniVL_audio '
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--audio_transfer', action='store_true',
                   help=('The task will transfer videos to audios that can be read by soundfile'))
    p.add_argument('--build_audio_statistics', action='store_true',
                   help=('The task will build audio statistics for audios in audio_dir'))
    p.add_argument('--build_audio_csv', action='store_true',
                   help=('The task will build a test csv '))
    p.add_argument('--build_pkl_subset', action='store_true',
                   help=('The task will build a caption subset '))
    p.add_argument('--build_annotation_statistics', action='store_true',
                   help=('The task will build annotation statistics '))
    p.add_argument('--build_annotation', action='store_true',
                   help=('The task will build annotation '))
    p.add_argument('--csvs_combine', action='store_true',
                   help=('Use meta data to rename the audios'))
    p.add_argument('--csvs_split', action='store_true',
                   help=('split csv to save a subset for validation'))
    p.add_argument('--csv2lst', action='store_true',
                   help=('build lst from video csv'))
    p.add_argument('--np64tonp16', action='store_true',
                   help=('transfer double numpy to float16 from npy_dir to new_npy_dir and delete original npy'))
    p.add_argument('--generate_video_segments', action='store_true',
                   help=('transfer double numpy to float16 from npy_dir to new_npy_dir and delete original npy'))
    p.add_argument('--audio_csv_minus', action='store_true',
                   help=('csv1 - csv2'))
    p.add_argument('--load_video_into_frames', action='store_true',
                   help=('transfer video  into frames'))
    p.add_argument('--build_catalog', action='store_true',
                   help=('transfer video  into Hierarchical directory'))

    p.add_argument('--video_dir', type=str, default="../data/audiocaps/videos_fps3",
                   help=('video dir'))
    p.add_argument('--new_video_dir', type=str, default="/dataset/28d47491/rld/data/youcookii/video_segments",
                   help=('new video dir'))
    p.add_argument('--input_dir', type=str, default="../data/Howto100m/audios_16k",
                   help=('video dir'))
    p.add_argument('--output_dir', type=str, default="../data/Howto100m/audios_cate",
                   help=('new video dir'))
    p.add_argument('--audio_dir', type=str, default="../data/audioset/audios_16k",
                   help='Output directory where audios to be saved')
    p.add_argument('--npy_dir', type=str, default="../data/audiocaps/raw_frames",
                   help=('npy dir'))
    p.add_argument('--new_npy_dir', type=str, default="./data/clotho/audio_feature_2channel_float16",
                   help='Output directory where new npy to be saved')

    p.add_argument('--input_csv', type=str, default=None,
                   help=('original csv file containing video_ids'))
    p.add_argument('--output_csv', type=str, default="data/activitynet/train_audio.csv",
                   help=('new csv file containing video_ids with paired audios'))

    p.add_argument('--input_pkl', type=str, default=None,
                   help=('annotaion file for caption'))
    p.add_argument('--output_pkl', type=str, default=None,
                   help=('annotation file for final caption'))

                   
    p.add_argument('--sample_num', type=int, default=None,
                   help=('sample number of videos with audio'))
    p.add_argument('--output_json', type=str, default="../data/audioset/audioset_transfer_statics.json",
                   help=('new json file to save basic audio information'))     
    p.add_argument('--annotation_path', type=str, default="./data/clotho/caption.pickle",
                   help=('annotation file'))            
    p.add_argument('-n', '--num_jobs', type=int, default=100)
    args= p.parse_args()

    if args.csvs_combine:
        assert args.input_csv is not None and args.output_csv is not None 
        input_csvs= args.input_csv.split(',')
        
        csvs_combine(input_csvs, args.output_csv)

    if args.csvs_split:
        assert args.input_csv is not None and args.output_csv is not None 
        output_csvs= args.output_csv.split(',')
        csvs_split(args.input_csv, output_csvs)

    if args.build_pkl_subset:
        assert args.input_csv is not None and args.input_pkl is not None and args.output_pkl
        build_pkl_subset(args.input_csv,args.input_pkl, args.output_pkl)


    if args.audio_transfer:
        '''
        python data_processor.py --audio_transfer --video_dir ../data/Howto100m/audios \
        --audio_dir ../data/Howto100m/audios_cate --output_json ../data/Howto100m/audio_transfer.json \
        --input_csv data/howto100m/HOWTO100M_audio_matched.csv -n 1
        '''
        assert args.audio_dir is not None and args.output_json is not None and args.video_dir is not None 
        # if os.path.exists(args.audio_dir):
        #     os.system(' '.join(['rm', '-r', args.audio_dir]))
        #     print('delete old file...')
        audio_transfer(args.video_dir, args.audio_dir, args.input_csv, args.output_json, args.num_jobs)
    
    if args.build_audio_statistics:
        assert args.audio_dir is not None and args.output_json is not None
        build_audio_statistics(args.audio_dir, args.output_json, args.sample_num)
    
    if args.build_audio_csv:
        '''
        python data_processor.py --build_audio_csv --input_csv ../data/activitynet/val_1.json 
        --output_csv data/activitynet/val_1.csv --audio_dir None
        '''
        assert args.input_csv is not None and args.output_csv is not None 
        build_audio_csv(args.input_csv, args.output_csv, args.audio_dir, args.video_dir, args.sample_num, args.num_jobs)
    
    if args.audio_csv_minus:
        assert args.input_csv is not None and args.output_csv is not None
        original_csv, minus_csv = arg.input_csv.split(',')
        audio_csv_minus(original_csv, minus_csv, args.output_csv)

    if args.build_annotation_statistics:
        build_annotation_statistics(args.audio_dir, args.annotation_path, args.output_json)
    
    if args.np64tonp16:
        assert args.npy_dir is not None and args.new_npy_dir is not None
        if not os.path.exists(args.new_npy_dir):
            os.mkdir(args.new_npy_dir)
        np64tonp16(args.npy_dir, args.new_npy_dir, args.input_csv, args.num_jobs)
    
    if args.csv2lst:
        assert args.input_csv is not None and args.output_csv is not None and args.npy_dir is not None
        csv2lst(args.input_csv, args.output_csv, args.npy_dir)
    
    if args.generate_video_segments:
        assert args.video_dir is not None and args.new_video_dir is not None and args.input_csv is not None
        if not os.path.exists(args.new_video_dir):
            os.mkdir(args.new_video_dir)
        generate_video_segments(args.video_dir, args.new_video_dir, args.input_csv)
    
    if args.build_annotation:
        '''
        python data_processor.py --build_annotation --input_csv ../data/vatex/vatex_training_v1.0.json,../data/vatex/vatex_validation_v1.0.json,../data/vatex/vatex_public_test_english_v1.1.json
        --output_pkl data/vatex/caption.pickle
        '''
        assert args.input_csv is not None  and  args.output_pkl is not None
        input_csvs = args.input_csv.split(',') 
        build_annotation(input_csvs, args.output_pkl, args.input_pkl)
    if args.load_video_into_frames:
        '''
        python data_processor.py  --load_video_into_frames --video_dir /dataset/8219588f/howto100m/videos --npy_dir ../data/Howto100m/raw_frames --input_csv ./data/howto100m/HOWTO100M.csv -n 72 
        '''
        assert args.video_dir is not None and args.npy_dir is not None and args.num_jobs is not None
        if not os.path.exists(args.npy_dir):
            os.mkdir(args.npy_dir)
        load_video_into_frames(args.video_dir, args.npy_dir, args.input_csv, args.num_jobs)
    
    if args.build_catalog:
        '''
        python data_processor.py  --build_catalog --input_dir ../data/Howto100m/audios_16k --output_dir ../data/Howto100m/audios_16k_cate --input_csv data/howto100m/HOWTO100M_10w.csv -n 64
        '''
        assert args.input_dir is not None and args.output_dir is not None and args.input_csv is not None
        build_catalog(args.input_dir, args.output_dir, args.input_csv, args.num_jobs)