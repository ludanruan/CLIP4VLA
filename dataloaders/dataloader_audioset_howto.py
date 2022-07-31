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

class Audioset_Howto_DataLoader(Dataset):
    """
    Audioset and Howto dataset loader.
    combine 2 csv, use a random number to decide which one chose,
    Note: Use transcript as caption, for mask decoder pretrain task.
    """

    def __init__(
            self,
            csv,
            features_path,
            data_dict, # audioset
            tokenizer,
            min_time=10.0,
            feature_framerate=1.0,
            max_words=30,
            min_words=0,
            n_pair=-1,          # every n_pair segments are considered as a sample
            max_frames=6,
            with_long_context=True,
            use_mil=False,
            only_sim=False,     # set automatically from model choice
            sampled_use_mil=False, # sample one sample from a video, not all segments in each video is used
            pretrain_enhance_vmodal=False,
            video_dim=1024,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,

            audio_path=None,
            max_audio_length = 6,
            audio_tokenlen = 1,
            audio_channel = 2,
            audio_rate = 16000,
            audio_overlap=0,
            audio_resolution = 224,
            with_decoder=True,
            enhance_single_modal = 0, 

            use_lmdb=False,
            feat_db=None
            
    ):
        """
        Args:
        """
        self.csv = {'audioset':pd.read_csv(csv['audioset']), 'howto100m':pd.read_csv(csv['howto100m'])}#pd.DataFrame(columns=["video_id"])#              
       
        self.features_path = features_path
        
        self.data_dict = data_dict
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.min_words = min_words
        self.tokenizer = tokenizer
        self.n_pair = n_pair
        self.with_long_context = with_long_context
        self.feature_size = video_dim
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
        self.iter_num = len(self.csv['audioset']) +len(self.csv['audioset']) 

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

        self.use_lmdb = use_lmdb
        if self.use_lmdb:
            self.feat_db = feat_db
        
    
    def __len__(self):
        
        return self.iter_num

    def _mask_tokens(self, words, enhance_vmodal=False):
        token_labels = []
        masked_tokens = words.copy()
        
        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            if enhance_vmodal:
                masked_tokens[token_id] = self.SPECIAL_TOKEN["MASK_TOKEN"]
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

    def _get_text(self, video_id, n_pair_max, sub_ids=None, only_sim=False, enhance_vmodel=False, text_ladder_mask=False):
        data_dict = self.data_dict[self.data_choose][video_id]
  
        if sub_ids is not None:
            k = len(sub_ids)
            r_ind = sub_ids
        else:
            n_caption = len(data_dict[self.data_choose]['start'])
            if n_pair_max == -1:
                k = n_caption
                r_ind = range(n_caption)
            else:
                k = n_pair_max
                if k <= n_caption:
                    r_ind = np.random.choice(range(n_caption), k, replace=False)
                else:
                    r_ind_must = np.array(range(n_caption))
                    r_ind_rand = np.random.choice(range(n_caption), k-n_caption, replace=True)
                    r_ind = np.concatenate((r_ind_must, r_ind_rand), axis=0)
                np.random.shuffle(r_ind)

        starts = np.zeros(k)
        ends = np.zeros(k)
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
            ind = r_ind[i]
            words, start_, end_ = self._get_single_transcript(data_dict, ind, with_long_context=self.with_long_context)
            caption_words = words.copy()
            
            starts[i], ends[i] = start_, end_

            # if enhance_vmodel:
            #     words = []      # mask all input text

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

                masked_tokens, token_labels = self._mask_tokens(words, enhance_vmodel)
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
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends 
        else:
            return pairs_text, pairs_mask, ladder_masks, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends

    def _get_single_transcript(self, data_dict, ind, with_long_context=True):
        start, end = ind, ind
        caption = str(data_dict['text'][ind])
        if self.data_choose == 'audioset':
            caption = 'The sound of ' + caption
      
        words = self.tokenizer.tokenize(caption)
        diff = data_dict['end'][end] - data_dict['start'][start]
        while with_long_context and (len(words) < self.min_words or diff < self.min_time):
            '''when time_duration < min_time  or len(text)<self.min_words, extend start time and end time'''
            if start > 0 and end < len(data_dict['end']) - 1:
                next_words = self.tokenizer.tokenize(str(data_dict['text'][end + 1]))
                prev_words = self.tokenizer.tokenize(str(data_dict['text'][start - 1]))
                d1 = data_dict['end'][end + 1] - data_dict['start'][start]
                d2 = data_dict['end'][end] - data_dict['start'][start - 1]
                if (self.min_time > 0 and d2 <= d1) or \
                    (self.min_time == 0 and len(next_words) <= len(prev_words)):
                    start -= 1
                    words = prev_words + words
                else:
                    end += 1
                    words.extend(next_words)
            elif start > 0:
                words = self.tokenizer.tokenize(str(data_dict['text'][start - 1])) + words
                start -= 1
            elif end < len(data_dict['end']) - 1:
                words.extend(self.tokenizer.tokenize(str(data_dict['text'][end + 1])))
                end += 1
            else:
                break
            diff = data_dict['end'][end] - data_dict['start'][start]
        return words, data_dict['start'][start], data_dict['end'][end]
    
    def _expand_video_slice(self, s, e, si, ei, fps, video_features):
        #not used
        start = int(s[si] * fps)
        end = int(e[ei] * fps) + 1

        if start > end:
            start, end = end, start
        video_slice = video_features[start:end]

        expand_left = True
        while len(video_slice) < 1:
            if si==0 and ei==len(s)-1:
                break
            if expand_left:
                expand_left = False
                si = si-1 if si>0 else si
            else:
                expand_left = True
                ei = ei+1 if ei<len(e)-1 else ei
            start = int(s[si] * fps)
            end = int(e[ei] * fps) + 1
            if start > end:
                start, end = end, start
            video_slice = video_features[start:end]

        if self.max_frames < video_slice.shape[0]:
            video_slice = video_slice[:self.max_frames]

        return video_slice, start, end

    def _get_rawvideo(self, video_id, s, e, only_sim=False, is_mask=False):
       
        if isinstance(video_id, int):
            if self.data_choose == 'howto100m':
                video_name = self.csv[self.data_choose]["feature_file"][video_id][:-4]
                video_id = video_name.split('.')[0] 
            elif self.data_choose == 'audioset':
                video_id = self.csv[self.data_choose]["video_id"][video_id]

        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        video_labels_index = -1 * np.ones_like(video_mask, dtype=np.long)
        max_video_length = [0] * len(s)
        
        try:
            for i in range(len(s)):
                get_frame_on = time.time()
                if self.data_choose == 'howto100m':                    
                    start= math.floor(s[0])
                    end = math.ceil(e[0])
                    if 'raw_frames' in self.features_path[self.data_choose]:
                        frame_dir = os.path.join(self.features_path[self.data_choose], video_id[0], video_id[1], video_id)
                        raw_video_data = self.rawVideoExtractor.get_video_frames(frame_dir, start_time=start, end_time=end)    
                    else:
                        video_path = os.path.join(self.features_path[self.data_choose], video_name)
                        raw_video_data = self.rawVideoExtractor.get_video_data_for_pre(video_path, start_time=start, end_time= end, slice_framepos=self.slice_framepos, max_frames = self.max_frames)
                elif self.data_choose == 'audioset':
                                        
                    if 'raw_frames' in self.features_path[self.data_choose]:
                        frame_dir = os.path.join(self.features_path[self.data_choose], video_id[0], video_id[1], video_id)
                        raw_video_data = self.rawVideoExtractor.get_video_frames(frame_dir)
                    else:
                        video_path = os.path.join(self.features_path[self.data_choose], self.csv[self.data_choose]["video_id"][vid]+'.mp4')
                        raw_video_data = self.rawVideoExtractor.get_video_data_for_pre(video_path, slice_framepos=self.slice_framepos, max_frames = self.max_frames)
                get_frame_off =time.time() - get_frame_on
                if get_frame_off >10: print("get video frame slow: video_id:{} time:{}".format(video_id, get_frame_off))
                if len(raw_video_data.shape) > 3:
                    # L x T x 3 x H x W
                    video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)
                else:
                    video_slice = raw_video_data

                if len(video_slice) < 1:
                    raise ValueError("{} is empty.".format(video_path))


                if self.max_frames < video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = video_slice[sample_indx, ...]
                                          
                slice_shape = video_slice.shape
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
                if len(video_slice) < 1:
                    pass
                else:
                    video[i][:slice_shape[0]] = video_slice
        except Exception as e:
            logger.warning("dataset:{} video_id: {} error:{}".format(self.data_choose, video_id, e))
           
            masked_video =  video.copy()          
                
            return video, video_mask, masked_video, video_labels_index

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length if v_length > 0 else [1] * 1

        if is_mask:
            masked_video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
            video_labels_index = -2 * np.ones_like(video_mask, dtype=np.long)
            

        else:
            # Mask Frame Model <-----
            masked_video = video.copy()
            if only_sim is False:
                video_labels_index = [[] for _ in range(len(s))]
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


    def _get_rawaudio_frames(self, video_id, s, e, only_sim=False, is_mask=False):
        audio_mask = np.zeros((len(s), self.max_audio_length), dtype=np.long)
        bg_token_gt = np.ones(len(s), dtype=np.long) * 1
        audio = np.zeros((len(s), self.max_audio_length, 3,
                          self.audio_resolution, self.audio_resolution), dtype=np.float)
        masked_audio = audio.copy()
        max_audio_length = [0] * len(s)
        audio_labels_index = -1 * np.ones_like(audio_mask, dtype=np.long)

          
        if isinstance(video_id, int):
            video_id = self.csv[self.data_choose]["video_id"][video_id]
        if 'cate' in self.audio_path[self.data_choose]:
            audio_path=os.path.join(self.audio_path[self.data_choose],video_id[0], video_id[1], video_id+'.wav')
        else:
            audio_path = os.path.join(self.audio_path[self.data_choose], video_id+'.wav')
       
        try:
                    
            for i in range(len(s)): 
                if self.data_choose=='howto100m':     
                    start= math.floor(s[i])
                    end = math.ceil(e[i])
                    audio_wav =  get_raw_audio(audio_path, self.audio_rate,start, end)
                elif self.data_choose=='audioset':
                    audio_wav =  get_raw_audio(audio_path, self.audio_rate)
                
                # get_frame_on = time.time() 
                audio_frame_l = wav2fbank(audio_wav[:,0], self.audio_rate)
                audio_frame_r = wav2fbank(audio_wav[:,1], self.audio_rate)
                audio_frame_m = wav2fbank((audio_wav[:, 0] + audio_wav[:, 1])/2, self.audio_rate)
                audio_frame =  np.stack([audio_frame_l, audio_frame_m, audio_frame_r],axis=0)#np.repeat(np.expand_dims(audio_frame_m, 0), 3, 0)#
                # get_frame_off = time.time() - get_frame_on
                
                # if get_frame_off > 10: print('get audio frame {} over time:{}'.format(video_id, get_frame_off))
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
                max_audio_length[i] = max_audio_length[i] if max_audio_length[i] > slice_len else slice_len
                
                audio[i][:max_audio_length[i]] = audio_frame[:max_audio_length[i]]
                masked_audio[i][:max_audio_length[i]] = masked_audio_frame[:max_audio_length[i]]
                audio_mask[i][:max_audio_length[i]] = [1] * max_audio_length[i]
                bg_token_gt[i] = 1  if self.data_choose == 'audioset' else 0
                  
                    
        except Exception as e:
            '''
            return blank if there exists no audio
            '''
            logger.warning("dataset:{} audio_id: {} error:{} ".format(self.data_choose, video_id, e, traceback.format_exc()))
            
            
           
        return audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt

    def __getitem__(self, feature_idx):
        '''
        
        is_audio: 
            if true, audio part returns 0 (no file), wave data(rate 16000)
            if false, audio part returns None 
        '''
        
       
        prob = random.random()
        if feature_idx < len(self.csv['audioset']):
            self.data_choose = 'audioset'
        else:
            self.data_choose = 'howto100m'
            feature_idx = feature_idx % len(self.csv['howto100m'])

        if  self.sampled_use_mil:  # sample from each video, has a higher priority than use_mil. true
            '''randomly sampleled an video input(3 segment) from video of feature_isx'''
            
            idx = feature_idx
            video_id = self.csv[self.data_choose]['video_id'].values[idx]
            data_dict = self.data_dict[self.data_choose][video_id]
            
            n_caption = len(data_dict['start'])
            if n_caption < self.n_pair:
                ranint = 0
            else:
                ranint = np.random.randint(0, n_caption//self.n_pair)
            sub_ids = [ranint*self.n_pair+i % n_caption for i in range(self.n_pair)]
            
        else:
            assert NotImplementedError
                
        enhance_vmodel = False
        mask_visual_modal = False
        mask_audio_modal = False


        pairs_text, pairs_mask, ladder_masks, pairs_segment, \
            pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
            pairs_decoder_mask, pairs_output_caption_ids, \
            starts, ends = self._get_text(video_id, self.n_pair, sub_ids, only_sim=self.only_sim, enhance_vmodel=enhance_vmodel)
        
     
        get_video_on = time.time()
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(idx, starts, ends, only_sim=self.only_sim, is_mask=mask_visual_modal)
        get_video_off = time.time() - get_video_on
       
        if get_video_off > 10: print('video slow dataset:{} video_id:{} fetch_time:{}'.format(self.data_choose, video_id, get_video_off))
        
    
        get_audio_on = time.time()  
        audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt = self._get_rawaudio_frames(idx, starts, ends, only_sim=self.only_sim, is_mask = mask_audio_modal) 
        get_audio_off = time.time() - get_audio_on
        if get_audio_off > 10: print('audio slow dataset:{} video_id:{} fetch_time:{}'.format(self.data_choose, video_id, get_audio_off))
        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
            audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt
            
