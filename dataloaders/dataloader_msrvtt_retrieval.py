from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import pdb
import pandas as pd
import soundfile as sf
from collections import defaultdict
import json
import random
import math
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.data import SPECIAL_TOKEN_CLIP
from dataloaders.dataloader_base import Base_DataLoader
from tqdm import tqdm
from dataloaders.rawaudio_util import *



class MSRVTT_Retrieval_DataLoader(Base_DataLoader):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
            
            frame_order=0,
            slice_framepos=0,
            audio_path=None,
            max_audio_length=12,
            audio_resolution=224,
            audio_tokenlen=1,  
            audio_channel=2,
            audio_rate=16000, 
            audio_overlap=0,
             
            video_path = None,
            filter_video_id = False
    ):
        super(MSRVTT_Retrieval_DataLoader, self).__init__()
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
        self.feature_dict = {}
        if os.path.isfile(self.features_path):
            self.feature_dict = pickle.load(open(self.features_path, 'rb'))
            self.feature_size = self.feature_dict[self.csv['video_id'].values[0]].shape[-1]
        sentences_dict = dict(zip(self.csv['video_id'].values,self.csv['sentence'].values))
        val_video_ids = list(self.csv['video_id'].values)
        val_video_ids = self.video_ids_filter(val_video_ids)
        self.video_id2idx=dict(zip(val_video_ids, list(range(len(val_video_ids)))))
        self.sentences_dict={}
        for video_id in val_video_ids:
            self.sentences_dict[len(self.sentences_dict)] = (video_id, sentences_dict[video_id])
    
    def __len__(self):
        return len(self.sentences_dict)
        
    def get_meta(self):
        meta = {"video_path":[],"captions":[]}
        for video_id, sentence in self.sentences_dict.values():
          
            meta["video_path"].append(os.path.join(self.video_path, video_id +'.mp4'))
            meta["captions"].append(sentence)
        
        return meta

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = self.SPECIAL_TOKEN["MASK_TOKEN"]
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["UNK_TOKEN"]])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["PAD_TOKEN"]])
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, choice_video_ids

    def __getitem__(self, idx):
        video_id, sentence = self.sentences_dict[idx]
        video_idx = self.video_id2idx[video_id]
    
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, choice_video_ids = self._get_text(video_id, sentence)

 
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(video_id)
        
        audio, audio_mask, masked_audio, audio_labels_index  = self._get_rawaudio_frames(video_id)
        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               audio, audio_mask, masked_audio, audio_labels_index, video_idx 



class MSRVTT_Retrieval_TrainDataLoader(Base_DataLoader):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,

            unfold_sentences=False,
            
            audio_path=None,
            audio_overlap=0,
            max_audio_length=12,
            audio_tokenizer=None,
            audio_resolution=224, 
            audio_tokenlen=1,
            audio_rate=16000,
            audio_channel=2,

            filter_video_id=False      
    ):
        super(MSRVTT_Retrieval_TrainDataLoader, self).__init__()
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        
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
        self.audio_resolution = audio_resolution
        self.audio_rate = audio_rate
        self.audio_channel = audio_channel
        self.audio_tokenlen = audio_tokenlen
       
        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        self.filter_video_id = filter_video_id
        self.feature_dict = {}
      
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            train_video_ids = self.video_ids_filter(train_video_ids)
            self.video_id2idx=dict(zip(train_video_ids, list(range(len(train_video_ids)))))
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            train_video_ids = list(self.csv['video_id'].values)
            train_video_ids = self.video_ids_filter(train_video_ids)
            self.video_id2idx=dict(zip(train_video_ids, list(range(len(train_video_ids)))))
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences[itm['video_id']].append(itm['caption'])
                    num_sentences += 1
                    s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(train_video_ids)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = self.SPECIAL_TOKEN["MASK_TOKEN"]
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["UNK_TOKEN"]])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["PAD_TOKEN"]])
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words


    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        video_idx=self.video_id2idx[video_id]
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(video_id)
        audio, audio_mask, masked_audio, audio_labels_index  = self._get_rawaudio_frames(video_id)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               audio, audio_mask, masked_audio, audio_labels_index,video_idx   
