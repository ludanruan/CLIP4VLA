from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import soundfile as sf
import pdb
import time
import logging
import torch
import math
import librosa
import traceback
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.data import _compute_mask_indices
from dataloaders.data import SPECIAL_TOKEN_CLIP
from dataloaders.rawaudio_util import *

logger = logging.getLogger(__name__)

class Audioset_DataLoader(Dataset):
    """
    Youtube dataset loader.
    Note: Use transcript as caption, for mask decoder pretrain task.
    """

    def __init__(
            self,
            csv,
            features_path,
            data_dict,
            tokenizer,
            min_time=10.0,
            feature_framerate=1.0,
            max_words=30,
            min_words=0,
            n_pair=-1,          # every n_pair segments are considered as a sample
            max_frames=6,
    
            use_mil=False,
            only_sim=False,     # set automatically from model choice
            sampled_use_mil=False, # sample one sample from a video, not all segments in each video is used
            pretrain_enhance_vmodal=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,

            use_lmdb=False,
            feat_db=None,

            audio_path=None,
            max_audio_length = 10,
            audio_tokenlen = 1,
            audio_channel = 2,
            audio_rate = 16000,
            audio_overlap=0,
            audio_resolution = 224,
            with_decoder=True,
            enhance_single_modal = 0, 
            
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)              
       
        self.features_path = features_path
        self.use_lmdb = use_lmdb
        if self.use_lmdb:
            self.feat_db = feat_db

        self.data_dict = data_dict
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.min_words = min_words
        self.tokenizer = tokenizer
        self.n_pair = n_pair
        
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = SPECIAL_TOKEN_CLIP
        self.only_sim = only_sim
        self.pretrain_enhance_vmodal = pretrain_enhance_vmodal
        self.iter_num = len(self.csv)

        self.use_mil = use_mil
        self.sampled_use_mil = sampled_use_mil

        self.audio_path = audio_path
        self.max_audio_length = max_audio_length
        self.audio_tokenlen = audio_tokenlen
        self.audio_channel = audio_channel
        self.audio_rate = audio_rate
        self.audio_resolution = audio_resolution
        self.audio_overlap = audio_overlap

        self.with_decoder = with_decoder
        self.enhance_single_modal = enhance_single_modal
         
        if self.sampled_use_mil:        # sample from each video, has a higher priority than use_mil.
            self.use_mil = True
        
    
    def __len__(self):
        return self.iter_num

    def _mask_tokens(self, words):
        token_labels = []
        masked_tokens = words.copy()

        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_tokens[token_id] = self.SPECIAL_TOKEN["MASK_TOKEN"]
                elif prob < 0.9:
                    masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                try:
                    token_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    token_labels.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["UKN_TOKEN"]])
            else:
                token_labels.append(-1)

        return masked_tokens, token_labels

    def _get_text(self, video_id,  only_sim=False, enhance_vmodel=False, text_ladder_mask=False):
        data_dict = self.data_dict[video_id]
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)         
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)
        if self.with_decoder is False: ladder_masks = np.zeros((k, self.max_words, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            
            words = self._get_single_transcript(data_dict)
            caption_words = words.copy()

            if enhance_vmodel:
                words = []      # mask all input text

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < self.max_words:
                input_ids.append(self.tokenizer.vocab[self.SPECIAL_TOKEN["PAD_TOKEN"]])
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            
            input_mask = np.array(input_mask)
            if self.with_decoder is False:
                
                ladder_mask=np.repeat(np.expand_dims(input_mask, 1),self.max_words,axis=1)
                if not enhance_vmodel and text_ladder_mask: 
                    # enhance generation by generating ladder shaped mask
                    
                    ladder_mask = np.tril(input_mask)
                    #input_mask = input_mask[:,:,None,None] *  ladder_mask[None,None,:,:]               
                ladder_masks[i] = ladder_mask
            
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = input_mask
            pairs_segment[i] = np.array(segment_ids)

            if only_sim is False:
                # For generate captions
                 
                if self.with_decoder:
                    if len(caption_words) > total_length_with_CLS:
                        caption_words = caption_words[:total_length_with_CLS]
                    input_caption_words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words
                    output_caption_words = caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
                else:
                    if len(caption_words) > total_length_with_CLS-1:
                        caption_words = caption_words[:total_length_with_CLS-1]
                    input_caption_words = output_caption_words =[self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                masked_tokens, token_labels = self._mask_tokens(words)
                masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                masked_input_caption_words, input_token_labels = self._mask_tokens(input_caption_words)
                input_caption_words = masked_input_caption_words.copy()

                while len(masked_token_ids) < self.max_words:
                    masked_token_ids.append(0)
                    token_labels.append(-1)
                assert len(masked_token_ids) == self.max_words
                assert len(token_labels) == self.max_words

                # For generate captions
                input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
                output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
                decoder_mask = [1] * len(input_caption_ids)
                while len(input_caption_ids) < self.max_words:
                    input_caption_ids.append(0)
                    output_caption_ids.append(0)
                    decoder_mask.append(0)
                assert len(input_caption_ids) == self.max_words
                assert len(output_caption_ids) == self.max_words
                assert len(decoder_mask) == self.max_words

                pairs_masked_text[i] = np.array(masked_token_ids)
                pairs_token_labels[i] = np.array(token_labels)

                pairs_input_caption_ids[i] = np.array(input_caption_ids)
                pairs_output_caption_ids[i] = np.array(output_caption_ids)
                pairs_decoder_mask[i] = np.array(decoder_mask)
        
        if self.with_decoder:
            return pairs_text, pairs_mask,  pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids
        else:
            return pairs_text, pairs_mask, ladder_masks, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids

    def _get_single_transcript(self, data_dict):
        caption = 'The sound of ' + str(data_dict['text'][0])
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, video_id,  only_sim=False, is_mask=False):

        if isinstance(video_id, int):
            video_id = self.csv["video_id"][video_id]

        
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
     
        max_video_length = [0] * 1
        video_labels_index = -1 * np.ones_like(video_mask, dtype=np.long)
        
        
        try:
            if 'raw_frames' in self.features_path:
                frame_dir = os.path.join(self.features_path, video_id[0], video_id[1], video_id)
                raw_video_data = self.rawVideoExtractor.get_video_frames(frame_dir)
            else:
                video_path = os.path.join(self.features_path, video_id+'.mp4')
                raw_video_data = self.rawVideoExtractor.get_video_data_for_pre(video_path, slice_framepos = self.slice_framepos, max_frames = self.max_frames)                   
            
            if len(raw_video_data.shape) > 3:
                    # L x T x 3 x H x W
                video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)
            else:
                video_slice = raw_video_data
  
                    
            if self.max_frames < video_slice.shape[0]:
                if self.slice_framepos == 0:
                    video_slice = video_slice[:self.max_frames, ...]
                elif self.slice_framepos == 1:
                    video_slice = video_slice[-self.max_frames:, ...]
                else:
                    sample_indx = np.linspace(0, video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = video_slice[sample_indx, ...]
                
            slice_shape = video_slice.shape
            max_video_length[0] = max_video_length[0] if max_video_length[0] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                pass
            else:
                video[0][:slice_shape[0]] = video_slice
        except Exception as e:
            print("video_id: {} error:{} ".format(video_id, e))
            masked_video =  video.copy()
            return video, video_mask, masked_video, video_labels_index


        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length if v_length > 0 else [1] * 1
      
        # Mask Frame Model <-----
        
        masked_video = video.copy()
        if only_sim is False:
            video_labels_index = [[] for _ in range(1)]
            for i, video_pair_ in enumerate(masked_video):
                for j, _ in enumerate(video_pair_):
                    if j < max_video_length[i]:
                        prob = random.random()
                        # mask token with 15% probability
                        if prob < 0.15:
                            masked_video[i][j] = [0.] * video.shape[-1]
                            video_labels_index[i].append(j)
                            # label 表示第几帧是被mask的
                        else:
                            video_labels_index[i].append(-1)
                    else:
                        video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def _get_rawaudio_frames(self, video_id, only_sim=False, is_mask=False):
        audio_mask = np.zeros((1, self.max_audio_length), dtype=np.long)
        bg_token_gt = np.ones(1, dtype=np.long) * 1
        audio = np.zeros((1, self.max_audio_length, 3,
                          self.audio_resolution, self.audio_resolution), dtype=np.float)
        masked_audio = audio.copy()
        max_audio_length = [0] * 1
        audio_labels_index = -1 * np.ones_like(audio_mask, dtype=np.long)

          
        if isinstance(video_id, int):
            video_id = self.csv["video_id"][video_id]
        if 'cate' in self.audio_path:
            audio_path=os.path.join(self.audio_path,video_id[0], video_id[1], video_id+'.wav')
        else:
            audio_path = os.path.join(self.audio_path, video_id+'.wav')
       
        try:
                
            audio_wav =  get_raw_audio(audio_path, self.audio_rate)            
            get_frame_on = time.time() 
            audio_frame_l = wav2fbank(audio_wav[:,0], self.audio_rate)
            audio_frame_r = wav2fbank(audio_wav[:,1], self.audio_rate)
            audio_frame_m = wav2fbank((audio_wav[:, 0] + audio_wav[:, 1])/2, self.audio_rate)
            audio_frame =  np.stack([audio_frame_l, audio_frame_m, audio_frame_r],axis=0)#np.repeat(np.expand_dims(audio_frame_m, 0), 3, 0)#
            get_frame_off = time.time() - get_frame_on
            
            if get_frame_off > 10: print('get audio frame {} over time:{}'.format(video_id, get_frame_off))
            masked_audio_frame = audio_frame.copy()
            masked_audio_frame = spec_augment(masked_audio_frame)
            masked_audio_frame = split_frame(masked_audio_frame, overlap=self.audio_overlap, single_frame_len=self.audio_resolution)
            masked_audio_frame = audio_processor(masked_audio_frame)

            audio_frame = split_frame(audio_frame, overlap=self.audio_overlap, single_frame_len=self.audio_resolution)
            audio_frame = audio_processor(audio_frame)

            #[tokens_num, channel,tokenlen]
            if self.max_audio_length < audio_frame.shape[0]:
                # the way deal with long audio keeps the same with video
                if self.slice_framepos == 0:
                    audio_frame = audio_frame[:self.max_audio_length, ...]
                    masked_audio_frame = masked_audio_frame[:self.max_audio_length, ...]
                elif self.slice_framepos == 1:
                    audio_frame = audio_frame[-self.max_audio_length:, ...]
                    masked_audio_frame = masked_audio_frame[-self.max_audio_length:, ...]
                else:
                    start = int((audio_frame.shape[0] - self.max_audio_length)/2)
                    end = start + self.max_audio_length
                    audio_frame = audio_frame[start:end, ...]
                    masked_audio_frame = masked_audio_frame[start:end]
                
            slice_len = audio_frame.shape[0]
            max_audio_length[0] = max_audio_length[0] if max_audio_length[0] > slice_len else slice_len
               
            audio[0][:max_audio_length[0]] = audio_frame[:max_audio_length[0]]
            masked_audio[0][:max_audio_length[0]] = masked_audio_frame[:max_audio_length[0]]
            audio_mask[0][:max_audio_length[0]] = [1] * max_audio_length[0]
            bg_token_gt[0] = 1  
                  
                    
        except Exception as e:
            '''
            return blank if there exists no audio
            '''
            logger.warning("audio_id: {} error:{} ".format(video_id, e, traceback.format_exc()))
            
            
           
        return audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt

    def __getitem__(self, feature_idx):
        '''
        is_audio: 
            if true, audio part returns 0 (no file), wave data(rate 16000)
            if false, audio part returns None 
        '''
        
        video_id = self.csv['video_id'].values[feature_idx]
        data_dict = self.data_dict[video_id]
        enhance_vmodel = False
        mask_visual_modal = False
        mask_audio_modal = False
        assert self.enhance_single_modal < 0.425, "enhance_single_modal is set too large"
        
        if self.only_sim is False: 
            '''stage 2 '''
            prob = random.random()
            if self.pretrain_enhance_vmodal and prob < 0.15:
                 # mask all text by rate 0.15
                enhance_vmodel = True
            if prob >= 0.15 and prob < (0.15 + self.enhance_single_modal):
                mask_visual_modal = True
            elif prob >= (0.15 + self.enhance_single_modal) and prob < (0.15 + self.enhance_single_modal*2):
                mask_audio_modal = True
        
        pairs_text, pairs_mask, ladder_masks, pairs_segment, \
        pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
        pairs_decoder_mask, pairs_output_caption_ids = self._get_text(video_id,  only_sim=self.only_sim, enhance_vmodel=enhance_vmodel)
    
        
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(video_id, only_sim=self.only_sim, is_mask=mask_visual_modal)
            
        audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt = self._get_rawaudio_frames(video_id,  only_sim=self.only_sim, is_mask = mask_audio_modal) 
        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
            audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt
            
