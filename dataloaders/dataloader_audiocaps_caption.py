from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os, pdb
import numpy as np
import pickle
import random
import math
import soundfile as sf
from collections import defaultdict
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.data import SPECIAL_TOKEN_CLIP
from dataloaders.dataloader_base import Base_DataLoader
import torch

class Audiocaps_Caption_DataLoader(Base_DataLoader):
    """Audiocaps dataset loader."""
    def __init__(
            self,
            csv,
            caption_path,
            features_path,
            tokenizer,
            
            feature_framerate=1.0,
            max_words=32,
            max_frames=10,
            image_resolution=224,
            frame_order=0,
            slice_framepos=2,
            
            audio_path=None,
            max_audio_length = 10,
            audio_tokenlen = 1,
            audio_resolution = 224,
            audio_overlap =0,
            audio_rate = 16000,
            audio_channel = 2,
            video_path = None,
            filter_video_id = False
    ):
        """
        Args:
        """
        super(Audiocaps_Caption_DataLoader, self).__init__()
        self.csv = pd.read_csv(csv)

        self.caption_dict = pd.read_csv(caption_path)
      
        
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
        self.audio_overlap = audio_overlap
        self.audio_resolution = audio_resolution
        self.audio_rate = audio_rate
        self.audio_channel = audio_channel

        self.video_path = video_path

        self.filter_video_id = filter_video_id
        


        # Get iterator video ids
        video_id_list = [str(itm) for itm in self.csv['audio_id'].values]
        video_id_list = self.video_ids_filter(video_id_list)
        #self.video_id2id_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
        self.data_dict ={video_id:{} for video_id in video_id_list}
        
        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        # Get all captions
        self.meta = {"video_path":[], "captions":[]}
        
        for idx, video_id in enumerate(video_id_list):
            
            caption_indexs = self.caption_dict[self.caption_dict['youtube_id']==video_id].index
            
            #training set one sentence, evaluate, all sentence
            
            for idy, caption_index in enumerate(caption_indexs):
                caption = self.caption_dict.loc[caption_index]['caption']
                
                self.video_sentences_dict[video_id].append(self.retokenized(caption, self.tokenizer))
                if 'train' in csv:
                    self.sentences_dict[len(self.sentences_dict)] = (video_id, caption)
                else:
                    if idy == 0: # training set 1 sentence/video, 
                        self.sentences_dict[len(self.sentences_dict)] = (video_id, caption)
            
            # self.data_dict[video_id]={'file_name':video_id,  'caption':caption}
            
            self.meta["video_path"].append(os.path.join(self.video_path, video_id + '.mp4')) 
            self.meta["captions"].append(str(self.video_sentences_dict[video_id]))
           
   

    def __len__(self):
        return len(self.sentences_dict)

    def get_meta(self):
        
        return self.meta

    def _get_text(self, video_id, caption=None):
        #data_dict = self.video_sentences_dict[video_id]
        k = 1

        starts = np.zeros(k)
        ends = np.zeros(k)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            
            words = []
            
            #start_, end_ = data_dict['start_end'][1:-1].split(', ') if data_dict['start_end'] != None else [0,0]
            #starts[i], ends[i] = math.floor(int(start_)/44100), math.ceil(int(end_)/44100)

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
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
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

            # For generate captions
            
            caption_words = self.tokenizer.tokenize(caption)
            
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words
            output_caption_words = caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            
            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["PAD_TOKEN"]])
                output_caption_ids.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["PAD_TOKEN"]])
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids
       
    def __getitem__(self, feature_idx):
        video_id, caption= self.sentences_dict[feature_idx]
        
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = self._get_text(video_id, caption)
  
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(video_id)
           
        audio, audio_mask, masked_audio, audio_labels_index  = self._get_rawaudio_frames(video_id)
         
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               audio, audio_mask, masked_audio, audio_labels_index,
