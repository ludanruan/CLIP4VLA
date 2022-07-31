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

class Youtube_DataLoader(Dataset):
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
            max_frames=12,
            with_long_context=True,
            use_mil=False,
            only_sim=False,     # set automatically from model choice
            sampled_use_mil=False, # sample one sample from a video, not all segments in each video is used
            pretrain_enhance_vmodal=False,
            video_dim=1024,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,

            use_lmdb=False,
            feat_db=None,

            audio_model='esresnet',
            audio_path=None,
            model_type = "audioclip",
            max_audio_length = 6,
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
        self.model_type = model_type
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
        self.iter_num = len(self.csv) * 100

        self.use_mil = use_mil
        self.sampled_use_mil = sampled_use_mil

        self.audio_model = audio_model
        self.audio_path = audio_path
        self.max_audio_length = max_audio_length
        self.audio_tokenlen = audio_tokenlen
        self.audio_channel = audio_channel
        self.audio_rate=audio_rate
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

    def _get_text(self, video_id, n_pair_max, sub_ids=None, only_sim=False, enhance_vmodel=False, text_ladder_mask=False):
        data_dict = self.data_dict[video_id]
  
        if sub_ids is not None:
            k = len(sub_ids)
            r_ind = sub_ids
        else:
            n_caption = len(data_dict['start'])
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
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends 
        else:
            return pairs_text, pairs_mask, ladder_masks, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends

    def _get_single_transcript(self, data_dict, ind, with_long_context=True):
        start, end = ind, ind
        words = self.tokenizer.tokenize(str(data_dict['text'][ind]))
        diff = data_dict['end'][end] - data_dict['start'][start]
        while with_long_context and (len(words) < self.min_words or diff < self.min_time):
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

    def _get_rawvideo(self, video_id, s, e, only_sim=False, is_mask=False):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        
        max_video_length = [0] * len(s)
        video_labels_index = -1 * np.ones_like(video_mask, dtype=np.long)
        if isinstance(video_id, int):
            video_name = self.csv["feature_file"][video_id][:-4]
            video_id = video_name.split('.')[0]
        
        try:
            for i in range(len(s)):
                start= math.floor(s[i])
                end = math.ceil(e[i])
                if 'raw_frames' in self.features_path:
                    frame_dir = os.path.join(self.features_path, video_id[0], video_id[1], video_id)
                    raw_video_data = self.rawVideoExtractor.get_video_frames(frame_dir, start_time=start, end_time=end)    
                else:
                    video_path = os.path.join(self.features_path, video_id + '.mp4')
                    if not os.path.exists(video_path):
                        video_path = os.path.join(self.features_path, video_id + 'webm')
                    raw_video_data = self.rawVideoExtractor.get_video_data_for_pre(video_path, start_time=start, end_time= end, slice_framepos=self.slice_framepos, max_frames = self.max_frames)
                
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
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
                if len(video_slice) < 1:
                    pass
                else:
                    video[i][:slice_shape[0]] = video_slice

        except Exception as e:
            print("video_id: {} error:{} ".format(video_id, e))
            traceback.print_exc()
            masked_video =  video.copy()
                
            return video, video_mask, masked_video, video_labels_index
            


        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length if v_length > 0 else [1] * 1
      
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
   
    def _get_rawaudio_slices(self, video_id, s, e, only_sim=False, is_mask=False):
        audio_token_len = int(self.audio_rate * self.audio_tokenlen)
       
        audio_mask = np.zeros((len(s), self.max_audio_length), dtype=np.long)
        audio = np.zeros((len(s), self.max_audio_length, self.audio_channel, audio_token_len), dtype=np.float)
        audio_labels_index = -1 * np.ones_like(audio_mask, dtype=np.long)
        if is_mask:
            masked_audio =  audio.copy()
            audio_labels_index = -1 * np.ones_like(audio_mask, dtype=np.long)
            return audio, audio_mask, masked_audio, audio_labels_index

        max_audio_length = [0] * len(s)
        if isinstance(video_id, int):
            video_id = self.csv["video_id"][video_id]
        
         
        audio_path = os.path.join(self.audio_path, video_id+'.wav')
        
        try:
            
            audio_wav, audio_rate = sf.read(audio_path)
            assert  self.audio_rate == audio_rate
                #down sampling
              
            if audio_wav.ndim == 1 and self.model_type=='audioclip':
                audio_wav = audio_wav[:, np.newaxis]
            else:
                
                audio_wav = (audio_wav[:, 0:1]+audio_wav[:, 1:2])/2
                audio_wav = (audio_wav - np.mean(audio_wav)) / np.sqrt(np.var(audio_wav) + 1e-5)
                

            if len(audio_wav) < 1:
                raise ValueError("{} is empty.".format(audio_path))
            
            for i in range(len(s)):
                
                #self.audio_compressrate 50 每50个tokens 算成1s
                audio_slice, start, end = self._expand_audio_slice(s, e, i, i, self.audio_rate , audio_wav)
                audio_slice=audio_slice.T
                if self.model_type == 'audioclip':
                    audio_slice = audio_slice * 32768.0  
                
                audio_cut = audio_slice.shape[-1] % audio_token_len
                if audio_cut>0:
                    audio_slice = audio_slice[:,:-1*audio_cut] 
                audio_slice = audio_slice.transpose(1,0).reshape(-1, audio_token_len, audio_slice.shape[0]).transpose(0,2,1)
                
                if self.max_audio_length < audio_slice.shape[0]:
                    # the way deal with long audio keeps the same with video
                    if self.slice_framepos == 0:
                        audio_slice = audio_slice[:self.max_audio_length, ...]
                    elif self.slice_framepos == 1:
                        audio_slice = audio_slice[-self.max_audio_length:, ...]
                    else:
                        sample_indx = np.linspace(0, audio_slice.shape[0]-1, num=self.max_audio_length, dtype=int)
                        audio_slice = audio_slice[sample_indx, ...]
                
                if self.transform is not None:
                    audio_slice = self.transform(audio_slice).numpy()

                slice_len = audio_slice.shape[0]
                max_audio_length[i] = max_audio_length[i] if max_audio_length[i] > slice_len else slice_len
                if slice_len < 1: 
                    pass
                else:
                    audio[i][:max_audio_length[i]] = audio_slice[:max_audio_length[i]]
                    audio_mask[i][:max_audio_length[i]] = [1] * max_audio_length[i]
                    
                    
        except Exception as e:
            print("audio_id: {} error:{} ".format(video_id, e))
            traceback.print_exc()
            masked_audio =  audio.copy()
                            
            return audio, audio_mask, masked_audio, audio_labels_index
            

        # Mask Frame Model <-----
        # masked_audio, audio_labels_index = self._mask_audio(audio)
        masked_audio = audio.copy()#[s,]       
        if only_sim is False:
            audio_labels_index = [[] for _ in range(len(s))]
            for i, audio_pair_ in enumerate(masked_audio):
                for j, _ in enumerate(audio_pair_):
                    if j < max_audio_length[i]:
                        prob = random.random()
                        # mask token with 15% probability
                        if prob < 0.15:
                            masked_audio[i][j] = [0.] * audio.shape[-1]
                            audio_labels_index[i].append(j)
                            # label 表示第几帧是被mask的
                        else:
                            audio_labels_index[i].append(-1)
                    else:
                        audio_labels_index[i].append(-1)
            audio_labels_index = np.array(audio_labels_index, dtype=np.long)
        # -----> Mask Frame Model
        
        return audio, audio_mask, masked_audio, audio_labels_index

    def _get_rawaudio_frames(self, video_id, s, e, only_sim=False, is_mask=False):
     
        
        audio_mask = np.zeros((len(s), self.max_audio_length), dtype=np.long)
        bg_token_gt = np.ones((len(s), self.max_audio_length), dtype=np.long) * -1
        audio = np.zeros((len(s), self.max_audio_length, 3,
                          self.audio_resolution, self.audio_resolution), dtype=np.float)
        max_audio_length = [0] * len(s)
        audio_labels_index = -1 * np.ones_like(audio_mask, dtype=np.long)

       
        if isinstance(video_id, int):
            video_id = self.csv["video_id"][video_id]

        if 'cate' in self.audio_path:
            audio_path=os.path.join(self.audio_path,video_id[0], video_id[1], video_id+'.wav')
        else:
            audio_path = os.path.join(self.audio_path, video_id+'.wav')
       
        try:
                    
            for i in range(len(s)): 
                 
                start= math.floor(s[i])
                end = math.ceil(e[i])
                audio_wav =  get_raw_audio(audio_path, self.audio_rate,start, end)
                
                get_frame_on = time.time() 
                audio_frame_l = wav2fbank(audio_wav[:,0], self.audio_rate)
                audio_frame_r = wav2fbank(audio_wav[:,1], self.audio_rate)
                audio_frame_m = wav2fbank((audio_wav[:, 0] + audio_wav[:, 1])/2, self.audio_rate)
                audio_frame =  np.stack([audio_frame_l, audio_frame_m, audio_frame_r],axis=0)#np.repeat(np.expand_dims(audio_frame_m, 0), 3, 0)#
                get_frame_off = time.time() - get_frame_on
                
                if get_frame_off > 10: print('get audio frame {} over time:{}'.format(video_id, get_frame_off))

                process_frame_on = time.time() 
                audio_frame = split_frame(audio_frame, overlap=self.audio_overlap, single_frame_len=self.audio_resolution)
                audio_frame = audio_processor(audio_frame)
                process_frame_off = time.time() - process_frame_on
                if process_frame_off > 10: print('process audio frame {} over time:{}'.format(video_id, process_frame_off))

            
                
                #[tokens_num, channel,tokenlen]
                if self.max_audio_length < audio_frame.shape[0]:
                    # the way deal with long audio keeps the same with video
                    if self.slice_framepos == 0:
                        audio_frame = audio_frame[:self.max_audio_length, ...]
                    elif self.slice_framepos == 1:
                        audio_frame = audio_frame[-self.max_audio_length:, ...]
                    else:
                        start = int((audio_frame.shape[0] - self.max_audio_length)/2)
                        end = start + self.max_audio_length
                        audio_frame = audio_frame[start:end, ...]
                
                slice_len = audio_frame.shape[0]
                max_audio_length[i] = max_audio_length[i] if max_audio_length[i] > slice_len else slice_len
                
                audio[i][:max_audio_length[i]] = audio_frame[:max_audio_length[i]]
                audio_mask[i][:max_audio_length[i]] = [1] * max_audio_length[i]
                bg_token_gt[i][:max_audio_length[i]] = [0]* max_audio_length[i]
                  
                    
        except Exception as e:
            '''
            return blank if there exists no audio
            '''
            logger.warning("audio_id: {} error:{} ".format(video_id, e))
            traceback.print_exc()
           
            masked_audio =  audio.copy()
            
            
            return audio, audio_mask, masked_audio, audio_labels_index,bg_token_gt

        # Mask Frame Model <-----
        # masked_audio, audio_labels_index = self._mask_audio(audio)  
        if is_mask:
            masked_audio = np.zeros((len(s), self.max_audio_length, self.audio_channel, audio_token_len), dtype=np.float)
            
            
        else:
            masked_audio = audio.copy()#[s,]
            if only_sim is False:
                audio_labels_index = [[] for _ in range(len(s))]
                for i, audio_pair_ in enumerate(masked_audio):
                    for j, _ in enumerate(audio_pair_):
                        if j < max_audio_length[i]:
                            prob = random.random()
                            # mask token with 15% probability
                            if prob < 0.15:
                                masked_audio[i][j] = [0.] * audio.shape[-1]
                                audio_labels_index[i].append(j)
                                # label 表示第几帧是被mask的
                            else:
                                audio_labels_index[i].append(-1)
                        else:
                            audio_labels_index[i].append(-1)
                audio_labels_index = np.array(audio_labels_index, dtype=np.long)
            # -----> Mask Frame Model
        
        return audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt

    def __getitem__(self, feature_idx):
        '''
        is_audio: 
            if true, audio part returns 0 (no file), wave data(rate 16000)
            if false, audio part returns None 
        '''
        prob = random.random()
        
        if  self.sampled_use_mil:  # sample from each video, has a higher priority than use_mil. true
            '''randomly sampleled an video input(3 segment) from video of feature_isx'''
            
            idx = feature_idx % len(self.csv['video_id'])
            video_id = self.csv['video_id'].values[idx]
            data_dict = self.data_dict[video_id]
            
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
        
        if self.with_decoder:
            pairs_text, pairs_mask, pairs_segment, \
            pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
            pairs_decoder_mask, pairs_output_caption_ids, \
            starts, ends = self._get_text(video_id, self.n_pair, sub_ids, only_sim=self.only_sim, enhance_vmodel=enhance_vmodel)
            
        else:

            pairs_text, pairs_mask, ladder_masks, pairs_segment, \
            pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
            pairs_decoder_mask, pairs_output_caption_ids, \
            starts, ends = self._get_text(video_id, self.n_pair, sub_ids, only_sim=self.only_sim, enhance_vmodel=enhance_vmodel)
        
        if self.model_type == 'no_audio':
            video, video_mask, masked_video, video_labels_index = self._get_video(idx, starts, ends, only_sim=self.only_sim, is_mask=mask_visual_modal)
        
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids
        
       
        get_video_on = time.time()
        video, video_mask, masked_video, video_labels_index = self._get_rawvideo(idx, starts, ends, only_sim=self.only_sim, is_mask=mask_visual_modal)
        get_video_off = time.time() - get_video_on
       
        if get_video_off > 10: print('video slow  video_id:{} fetch_time:{}'.format( video_id, get_video_off))
        
    
        get_audio_on = time.time()  
        if self.audio_model == 'esresnet':    
            audio, audio_mask, masked_audio, audio_labels_index = self._get_rawaudio_slices(idx, starts, ends, only_sim=self.only_sim, is_mask = mask_audio_modal) 
        elif self.audio_model == 'audio-clip':
            audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt = self._get_rawaudio_frames(idx, starts, ends, only_sim=self.only_sim, is_mask = mask_audio_modal) 
        get_audio_off = time.time() - get_audio_on
       
        if get_audio_off > 10: print('audio slow video_id:{} fetch_time:{}'.format(video_id, get_audio_off))
        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
            audio, audio_mask, masked_audio, audio_labels_index, bg_token_gt
            
