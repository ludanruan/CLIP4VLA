from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import pickle
import re
import random
import io
import pdb
import math
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.data import SPECIAL_TOKEN_CLIP
from dataloaders.dataloader_base import Base_DataLoader

class Youcook_Caption_DataLoader(Base_DataLoader):
    """Youcook dataset loader."""
    def __init__(
            self,
            csv,
            data_path,
            features_path,
            tokenizer,
            feature_framerate=1.0,
            max_words=48,
            max_frames=12,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,

            audio_path=None,
            max_audio_length = 12,
            audio_overlap = 0,
            audio_resolution = 224, 
            audio_rate = 16000,
            audio_tokenlen=1, 
            video_path = None,
            filter_video_id = False
    ):
        """
        Args:
        """
        super(Youcook_Caption_DataLoader, self).__init__()
        
        self.csv = pd.read_csv(csv)
        self.data_dict = pickle.load(open(data_path, 'rb'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

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
        self.audio_tokenlen = audio_tokenlen
        self.audio_rate = audio_rate
        self.video_path = video_path
       
        self.filter_video_id = filter_video_id

        # Get iterator video ids
        video_id_list = [itm for itm in self.csv['video_id'].values]
        video_id_list = self.video_ids_filter(video_id_list)
        
        self.video_id2idx_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
        # Get all captions
        self.iter2video_pairs_dict = {}
        self.meta = {"video_path":[], "captions":[]}
        iter_idx_ = 0
        for video_id in video_id_list:
            data_dict = self.data_dict[video_id]
            n_caption = len(data_dict['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[iter_idx_] = (video_id, sub_id)
                iter_idx_ += 1
                # build meta dict
                s = str(data_dict['start'][sub_id])
                e = str(data_dict['end'][sub_id])
                video_path = os.path.join(self.video_path, '_'.join([video_id,s,e])+'.mp4')
                caption = data_dict['text'][sub_id]
                self.meta["video_path"].append(video_path)
                self.meta["captions"].append(caption)
        

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def get_meta(self):
        
        return self.meta

    def _get_text(self, video_id, sub_id):
        data_dict = self.data_dict[video_id]
        k = 1
        r_ind = [sub_id]

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
            ind = r_ind[i]
            start_, end_ = data_dict['start'][ind], data_dict['end'][ind]
            starts[i], ends[i] = start_, end_
            
            total_length_with_CLS = self.max_words - 1
           

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"],self.SPECIAL_TOKEN["SEP_TOKEN"]]# + words
            
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
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
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
            caption_words = self.tokenizer.tokenize(data_dict['text'][ind])
            
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

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels,\
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends
   
    def __getitem__(self, feature_idx):

        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        idx = self.video_id2idx_dict[video_id]
        
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
        pairs_decoder_mask, pairs_output_caption_ids, starts, ends = self._get_text(video_id, sub_id)  
        
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(video_id, starts, ends)
        audio, audio_mask, masked_audio, audio_labels_index  = self._get_rawaudio_frames(video_id, starts, ends)
        

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               audio, audio_mask, masked_audio, audio_labels_index   
        