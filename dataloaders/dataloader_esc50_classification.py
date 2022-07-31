from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import pdb
import pandas as pd
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.data import SPECIAL_TOKEN_CLIP
from dataloaders.dataloader_base import Base_DataLoader
from dataloaders.rawaudio_util import *



class Esc50_Classification_DataLoader(Base_DataLoader):
    """Esc50 dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path=None,
            tokenizer=None,
            max_words=32,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
            
            frame_order=0,
            slice_framepos=0,
            audio_path=None,
            max_audio_length=3,
            audio_resolution=224,
            audio_tokenlen=1,  
            audio_channel=2,
            audio_rate=16000, 
            audio_overlap=0,
            video_path = None,
            filter_video_id = False,
            split="1"
    ):
        super(Esc50_Classification_DataLoader, self).__init__()
        self.csv = pd.read_csv(csv_path)
        
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.features_path = features_path
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = SPECIAL_TOKEN_CLIP 

        self.audio_path = audio_path
        self.max_audio_length = max_audio_length
        self.audio_overlap = audio_overlap
        self.audio_tokenlen = audio_tokenlen
        self.audio_channel = audio_channel
        self.audio_rate = audio_rate
        self.audio_resolution = audio_resolution
        self.video_path = video_path
        self.filter_video_id = filter_video_id
        self.split = split

        # sentences_dict = dict(zip(self.csv['video_id'].values,self.csv['sentence'].values))
        # audio_names = list(self.csv['filename'].values)
        
        self.data_dict={}
        for _, row in self.csv.iterrows():
            file_name, fold, label, cate, _,_,_ = row
            audio_p = os.path.join(self.audio_path,file_name)
            if os.path.exists(audio_p) == False:
                continue
            
            if  self.split.find(str(fold))==-1:
                continue
            self.data_dict[len(self.data_dict)]={'audio_id': file_name.split('.')[0],\
                'target': int(label), 'cate': cate}
        
    def __len__(self):
        return len(self.data_dict)
        

    def __getitem__(self, idx):
        data_item = self.data_dict[idx]
        audio_id, label = data_item['audio_id'], data_item['target']
    
        
        # pairs_text, pairs_mask = self._get_text(audio_id, cate)

        
        audio, audio_mask, _, _  = self._get_rawaudio_frames(audio_id)
           
        return audio, audio_mask, label


