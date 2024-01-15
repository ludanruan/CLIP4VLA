from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import pdb
import os
import shutil
import pickle
from tqdm import tqdm
import time
import argparse
from modules.tokenization import SimpleTokenizer as ClipTokenizer
from modules.tokenization import  END_TOKEN
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import AudioClip
from modules.optimization import BertAdam
from torch.utils.data import DataLoader
from util import *
from dataloaders.dataloaders import *

from metrics import t2v_metrics as compute_metrics
from metrics import sim_rerank
from utils.visualizer import Visualizer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
torch.distributed.init_process_group(backend="nccl")

def get_args(description='CLIP4VLA on Retrieval Task'):
    args = get_parser(description)
    
    # Check paramenters
    args.task_type="retrieval"
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    # if args.datatype == 'audiocaps': 
    #     assert args.multi_sentence == 5, 'input in right multi_sentence for audiocaps' 
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = mylog(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):   
    if args.init_model:
        state_dict = torch.load(args.init_model, map_location='cpu')
        
        model_state_dict = state_dict.copy()
        if args.init_model.endswith('pt'):
            for key, val in state_dict.items():
                if key.find('audio')!=0 and "logit_scale_a" not in key:
                    new_key = "clip." + key
                    if "token_embedding" in key:
                        #expand mask token
                        mask_expand = torch.randn(1,val.size(1))
                        
                        model_state_dict[key] = torch.cat([val, mask_expand], dim=0)
                    
                    model_state_dict[new_key] = model_state_dict.pop(key)
                    
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    model = AudioClip.from_pretrained(state_dict=model_state_dict, task_config=args)
    
    
    model.to(device)
    return model

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
       
        model = AudioClip.from_pretrained(state_dict=model_state_dict, task_config=args)
       
        model.to(device)
    else:
        model = None
    return model



def extract_all_features(args, model, test_dataloader, device, n_gpu, meta=None):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    
    model.eval()
    
    all_vid = []
    idx2vid = {_id:vid for vid, _id in test_dataloader.dataset.video_id2idx.items()}
    with torch.no_grad():
        batch_sequence_output_list, batch_visual_output_list, batch_audio_output_list = [], [], []
        
        for bid, batch in enumerate(test_dataloader):
            
            '''save tva dencode features in batch_*_output_list with batches and batchlist'''
            batch = tuple(t.to(device) for t in batch)
            if len(batch)==14:
                input_ids, input_mask, segment_ids, video, video_mask, _, _, _, _,audio,audio_mask, _, _,video_id = batch
            else:
                input_ids, input_mask, segment_ids, video, video_mask, _, _, _, _, _, _, _, audio,audio_mask, _, _,video_id = batch
            
            for idx in video_id:
                idx = float(idx.detach().cpu())
                vid = idx2vid[idx] 
                all_vid.append(vid)

            #######
            ########get sequence_output
            #######
            sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask,shaped=False).cpu().detach()
            
            if args.train_sim_after_cross is False:
                sequence_output = sequence_output[torch.arange(input_ids.shape[0]),   (input_ids == END_TOKEN).nonzero(as_tuple=True)[-1]]     
            
            batch_sequence_output_list.append(sequence_output)
            
            
            # if args.audio_complement:
            #     audio, audio_mask = model.complement_audio(audio, audio_mask,sequence_output, input_ids )
            loader_start_, loader_end_ = bid*args.batch_size_val,  bid * args.batch_size_val + audio.size(0)
            modal_filtered = [i% args.multi_sentence==0 for i in range(loader_start_, loader_end_)]
            if sum(modal_filtered) > 0:
                audio = audio[modal_filtered].to(torch.float32)
                audio_mask = audio_mask[modal_filtered].to(torch.float32)
                audio_output, _ = model.get_audio_output(audio, audio_mask, shaped = False)
                audio_output = audio_output.cpu().detach()
                video = video[modal_filtered].to(torch.float32)
                video_mask = video_mask[modal_filtered].to(torch.float32)
                visual_output = model.get_visual_output( video, video_mask, shaped=False).cpu().detach()
                    
                batch_visual_output_list.append(visual_output)
                batch_audio_output_list.append(audio_output)
            
            print("{}/{}\r".format(bid, len(test_dataloader)), end="")       
    return batch_sequence_output_list, batch_visual_output_list, batch_audio_output_list, all_vid


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_retrieval_train, "val":dataloader_msrvtt_retrieval_test}
DATALOADER_DICT["vatex"] = {"train":dataloader_vatex_retrieval_train, "val":dataloader_vatex_retrieval_test}
def main():
    global logger
    
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    
    tokenizer = ClipTokenizer()
   
    model = init_model(args, device, n_gpu, args.local_rank)

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length, meta = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)

    if args.local_rank == 0:
        text_feature, visual_feature, audio_feature, vids = extract_all_features(args, model, test_dataloader, device, n_gpu, meta)
        # import pdb;pdb.set_trace()
        text_features = torch.cat(text_feature, dim=0).detach().cpu().numpy()
        visual_features = torch.cat(visual_feature, dim=0).detach().cpu().numpy()
        audio_features = torch.cat(audio_feature, dim=0).detach().cpu().numpy()
        # assert text_features.size(0) == visual_features.size(0) == audio_features.size(0) == len(vids)
        vids = np.array(vids)
        np.save(args.output_dir+'/text_features.npy', text_features)
        np.save(args.output_dir+'/visual_features.npy', visual_features)
        np.save(args.output_dir+'/audio_features.npy', audio_features)
        np.save(args.output_dir+'/vids.npy', vids)

if __name__ == "__main__":
    main()
