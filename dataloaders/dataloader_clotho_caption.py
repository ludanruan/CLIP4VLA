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
import torch

class Clotho_Caption_DataLoader(Dataset):
    """Clotho dataset loader."""
    def __init__(
            self,
            csv,
            caption_path,
            meta_path,
            tokenizer,
            feature_framerate=1.0,
            max_words=30,
            max_frames=10,
            
            audio_path=None,
            model_type = "audio_feature",
         
            max_audio_length = 800,
            audio_compressrate = 50,
            audio_tokenlen = 0.02,
            audio_tokenizer = None,
            audio_dim=512,
            video_path=None,
            with_decoder = True,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)

        self.caption_dict = pd.read_csv(caption_path)
        self.meta_dict = pd.read_csv(meta_path)
        
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.audio_path = audio_path
        self.model_type = model_type
      
        self.max_audio_length = max_audio_length
        self.audio_compressrate = audio_compressrate
        self.audio_tokenlen = audio_tokenlen
        self.audio_tokenizer = audio_tokenizer
        self.audio_dim = audio_dim

        self.video_path = video_path
        self.with_decoder = with_decoder

        assert self.model_type == "audio_feature"

        # Get iterator video ids
        video_id_list = [str(itm) for itm in self.csv['audio_id'].values]
        #self.video_id2id_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
        self.data_dict ={video_id:{} for video_id in enumerate(video_id_list)}
        # Get all captions

        self.meta = {"video_path":[], "captions":[]}
        self.iter2video_pairs_dict = {}
        iter_idx_ = 0
        for idx, video_id in enumerate(video_id_list):
            
            meta_index = self.meta_dict[self.meta_dict['sound_id']==video_id].index[0]
            video_name = self.meta_dict['file_name'].loc[meta_index]
            start_end = self.meta_dict['start_end_samples'].loc[meta_index]
                
            caption_index = self.caption_dict[self.caption_dict['file_name']==video_name].index[0]
            captions = self.caption_dict.loc[caption_index].values[1:].tolist()
            #只取标注的第一个句子作为caption annotation
            self.data_dict[video_id]={'file_name':video_name, 'start_end':start_end, 'caption':captions[0]}
            n_caption = len(captions)
            self.meta["video_path"].append(os.path.join(self.video_path, video_name)) 
            self.meta["captions"].append(captions[0])
            self.iter2video_pairs_dict[iter_idx_] = video_id
            iter_idx_ += 1
   

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def get_meta(self):
        
        return self.meta

    def _get_text(self, video_id):
        data_dict = self.data_dict[video_id]
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

            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

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
                        masked_tokens[token_id] = "[MASK]"

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
            caption_words = self.tokenizer.tokenize(data_dict['caption'])
            if self.with_decoder:
                if len(caption_words) > total_length_with_CLS:
                    caption_words = caption_words[:total_length_with_CLS]
                input_caption_words = ["[CLS]"] + caption_words
                output_caption_words = caption_words + ["[SEP]"]
            else:
                
                if len(caption_words) > total_length_with_CLS-1:
                    caption_words = caption_words[:total_length_with_CLS-1]
                    
                    
                output_caption_words =  ["[CLS]"] + caption_words + ["[SEP]"]
                input_caption_words = []
                masked_tokens = caption_words.copy()+["[SEP]"]
                for token_id, token in enumerate(masked_tokens):
                    
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_tokens[token_id] = "[MASK]"
                input_caption_words = ["[CLS]"]+ masked_tokens 
                

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

    def _get_video(self, idx, s=[1], e=[1]):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        video = np.zeros((len(s), self.max_frames, 1024), dtype=np.float)
        masked_video = np.zeros_like(video, dtype=np.float)
        video_labels_index = np.zeros_like(video_mask, dtype=np.long)
        
        return video, video_mask, masked_video, video_labels_index
    
    def _audio_sparse_sampling(self, audio_feature, audio_compressrate):
       
        audio_feature_resize = audio_feature[::audio_compressrate,:]#audio_feature_temp.sum(axis=1)/np.expand_dims(audio_mask.sum(axis=1), -1)
        
        return audio_feature_resize

    def _audio_compress(self, audio_feature, audio_compressrate):
   
        audio_len, audio_dim = audio_feature.shape
        
        audio_resize_len = math.ceil(audio_len*1.0 / audio_compressrate)
        audio_feature_temp = np.zeros((audio_resize_len * audio_compressrate, audio_dim))
       
        audio_feature_temp[:audio_len, :] =  audio_feature
        audio_mask = np.zeros(audio_resize_len*audio_compressrate)
        audio_mask[:audio_len] = 1

        audio_feature_temp = audio_feature_temp.reshape((audio_resize_len, audio_compressrate, audio_dim))
        audio_mask = audio_mask.reshape((audio_resize_len, audio_compressrate))

        audio_feature_resize = audio_feature_temp.sum(axis=1)/np.expand_dims(audio_mask.sum(axis=1), -1)

        return audio_feature_resize

    
    def _get_audio_feature(self, video_id, s=[1], e=[1], only_sim=False):
        assert self.max_audio_length%self.audio_compressrate == 0
                
        audio_mask = np.zeros((len(s), self.max_audio_length // self.audio_compressrate), dtype=np.long)
        max_audio_length = [self.max_audio_length // self.audio_compressrate] * len(s)
        audio = np.zeros((len(s), self.max_audio_length // self.audio_compressrate, self.audio_dim), dtype=np.float)
        
        
        video_name = self.data_dict[video_id]["file_name"][:-4]
           
        audio_path = os.path.join(self.audio_path, video_name+'.npy')
    
        try:
            audio_slice = np.load(audio_path)

            for i in range(len(s)):
                if len(audio_slice) < 1:
                    raise ValueError("{} is empty.".format(audio_path))
                #self.audio_compressrate 50 每50个tokens 算成1s
                
                #audio_slice, start, end = self._expand_audio_slice(s, e, i, i, self.audio_compressrate, audio_features)

                audio_slice = self._audio_sparse_sampling(audio_slice, self.audio_compressrate)
                slice_shape = audio_slice.shape
                max_audio_length[i] = max_audio_length[i] if max_audio_length[i] < slice_shape[0] else slice_shape[0]
                if len(audio_slice) < 1:
                    print("audio_slice error:audio_id: {}, start: {}, end: {}".format(video_id, start, end))
                    pass
                else:
                    audio[i][:max_audio_length[i]] = audio_slice[:max_audio_length[i]]
                    audio_mask[i][:max_audio_length[i]] = [1] * max_audio_length[i]
                    
        except Exception as e:
            print("audio_id: {} error.".format(audio_path))

        # Mask Frame Model <-----
        audio_labels_index = [[] for _ in range(len(s))]
        masked_audio = audio.copy()
        if only_sim is False:
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
    
    

       
    def __getitem__(self, feature_idx):
        video_id= self.iter2video_pairs_dict[feature_idx]
        
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = self._get_text(video_id)
        video, video_mask, masked_video, video_labels_index = self._get_video(video_id)
        
       
        
        audio, audio_mask, masked_audio, audio_labels_index = self._get_audio_feature(video_id)
        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               audio, audio_mask, masked_audio, audio_labels_index,
