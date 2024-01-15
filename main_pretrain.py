from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os, pdb
from collections import OrderedDict
from torch.cuda.amp import autocast as autocast
import pickle
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import AudioClip
from dataloaders.data import LMDBGroup, LMDBFeature
from dataloaders.dataloaders import *
from util import *
from utils import _visdom

from torch.cuda.amp import GradScaler
torch.distributed.init_process_group(backend="nccl")

global logger
# torch.autograd.set_detect_anomaly(True)
def get_args(description='CLIP4VLA on Pretrain'):
    args = get_parser(description)

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_pretrain or args.do_eval:
        raise ValueError("`do_pretrain or do_eval`  must be True.")
    assert args.datatype in ['howto100m', 'audioset', 'audioset+howto']
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps) # 
        
    args.checkpoint_model = '{}_{}_{}_{}.checkpoint'.format(args.checkpoint_model, args.bert_model, args.max_words, args.max_frames)

    return args

def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()

    rank = torch.distributed.get_rank()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
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

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, local_rank, type_name="", global_step=-1, optimizer=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)

    if global_step != -1 and optimizer is not None:
        state_dict = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'last_optimizer_state': convert_state_dict_type(optimizer.state_dict()),
        }
        checkpoint_model_file = os.path.join(args.output_dir, args.checkpoint_model)
        torch.save(state_dict, checkpoint_model_file)
        logger.info("Checkpoint is saved. use `load_checkpoint` to recovery it.")

    return output_model_file

def load_model(epoch, args, n_gpu, device, model, global_step=0, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))

    last_optim_state = None
    checkpoint_model_file = os.path.join(args.output_dir, args.checkpoint_model)
    if epoch == -1 and args.load_checkpoint and os.path.exists(checkpoint_model_file):
        checkpoint_state = torch.load(checkpoint_model_file, map_location='cpu')
        epoch = checkpoint_state['epoch']
        global_step = checkpoint_state['global_step']
        model_state_dict = checkpoint_state['model_state_dict']
        last_optim_state = checkpoint_state['last_optimizer_state']
        
        
        model = AudioClip.from_pretrained(state_dict=model_state_dict, task_config=args)

        model.to(device)
        if args.local_rank == 0:
            logger.info("Checkpoint loaded from %s", checkpoint_model_file)
    elif os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        model = AudioClip.from_pretrained(state_dict=model_state_dict, task_config=args)

        model.to(device)

    return epoch, global_step, last_optim_state, model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, scaler, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    fetch_time_start=time.time()
   
    for step, batch in enumerate(train_dataloader):
        
        #logger.warning("Cuda:{} Time featch a batch:{}".format(args.local_rank, time.time()-fetch_time_start)) 
      
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)  
            
        input_dict = {'input_ids':batch[0],'token_type_ids':batch[2], 'attention_mask':batch[1], 'video':batch[3], \
                'video_mask':batch[4], 'pairs_masked_text':batch[5], 'pairs_token_labels': batch[6], \
                'masked_video':batch[7], 'video_labels_index':batch[8], 'input_caption_ids': batch[9], \
                'decoder_mask': batch[10],  'output_caption_ids':batch[11],  'audio': batch[12], 'audio_mask': batch[13], \
                'masked_audio': batch[14], 'audio_labels_index':batch[15]      }
        if args.with_bg_token or args.with_control_token:
            input_dict['bg_token_gt'] = batch[16]
            
            
        # with torch.autograd.detect_anomaly():
        
        if args.fp16:
            with autocast():
                loss, loss_recoder = model(**input_dict)
        else:
            start_run =time.time()
            loss, loss_recoder = model(**input_dict)
            #logger.warning('finish run 1 step forward:{}'.format(time.time()-start_run))

        # if torch.isnan(loss):
        #     logger.warning('Loss has NAN:{}, break'.format(loss))         
        if n_gpu > 1:
            loss = loss.mean()
                   
        if args.gradient_accumulation_steps > 1:
            loss /=args.gradient_accumulation_steps

        # with torch.autograd.detect_anomaly():    
        if args.fp16:   
            scaler.scale(loss).backward() 
 
        else:
            loss.backward()
        
        total_loss += float(loss)
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            #pdb.set_trace()
            
            assert torch.isnan(sum([a.sum() for a in model.state_dict().values()]) ) == False, logger.warning("model's gradients has nan")

            if scheduler is not None:
                scheduler.step()
                
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
               
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            assert torch.isnan(sum([a.sum() for a in model.state_dict().values()])) == False, logger.warning("model's gradients has nan")
            optimizer.zero_grad()
            
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                
                loss_recoder.mean(args.gradient_accumulation_steps)
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                
                logger.info("Loss items: %s",loss_recoder.return_str())

                start_time = time.time()
        fetch_time_start=time.time()
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch_tva(args, model, test_dataloader, device, n_gpu, meta=None):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    
    model.eval()
  
    with torch.no_grad():
        batch_list_sequence, batch_list_visual, batch_list_audio = [], [], []
      
        batch_sequence_output_list, batch_visual_output_list, batch_audio_output_list = [], [], []
        
        for bid, batch in enumerate(test_dataloader):
            
            '''save tva dencode features in batch_*_output_list with batches and batchlist'''
            batch = tuple(t.to(device) for t in batch)
            
            input_ids, input_mask, segment_ids, video, video_mask, _, _, _, _,audio,audio_mask, _, _ = batch
            audio = audio.to(torch.float32)
            audio_mask = audio_mask.to(torch.float32)
                    
            audio_output = model.get_audio_output(audio, audio_mask, shaped = False).cpu().detach()
          
            sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask,shaped=False).cpu().detach()
            
            
            sequence_output = sequence_output[torch.arange(input_ids.shape[0]),  input_ids.argmax(dim=-1).squeeze(1)]
            
            visual_output = model.get_visual_output( video, video_mask, shaped=False).cpu().detach()
            
            batch_sequence_output_list.append(sequence_output)
            batch_visual_output_list.append(visual_output)
            batch_audio_output_list.append(audio_output)
            batch_list_sequence.append([input_ids.cpu().detach(), input_mask.cpu().detach(), segment_ids.cpu().detach()])
            batch_list_visual.append([video.cpu().detach(), video_mask.cpu().detach()])
            batch_list_audio.append([audio.cpu().detach(), audio_mask.cpu().detach()])
           
            print("{}/{}\r".format(bid, len(test_dataloader)), end="")       
        
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_sequence_splits = []
            batch_list_visual_splits = []
            batch_list_audio_splits = []
            batch_sequence_output_splits = []
            batch_visual_output_splits = []
            batch_audio_output_splits = []
            batch_len = len(batch_list_sequence)
            split_len = (batch_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                 
                devc = torch.device('cuda:{}'.format(str(dev_id)))
                devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_sequence[s_:e_]]
                batch_list_sequence_splits.append(devc_batch_list)
                devc_batch_list = [b[1].to(devc) for b in batch_list_visual]
                batch_list_visual_splits.append(devc_batch_list)
                devc_batch_list = [b[1].to(devc) for b in batch_list_audio]
                batch_list_audio_splits.append(devc_batch_list)

                devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                batch_sequence_output_splits.append(devc_batch_list)
                devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                batch_visual_output_splits.append(devc_batch_list)
                devc_batch_list = [b.to(devc) for b in batch_audio_output_list]
                batch_audio_output_splits.append(devc_batch_list)


            sim_matrix = {'t_v':[],
                        't_a':[],
                        't_va':[],
            
                        'query_weights':[]}
            query_weights = []
            parameters_tuple_list = [(batch_list_sequence_splits[dev_id], batch_list_visual_splits[dev_id], batch_list_audio_splits[dev_id], 
                                      batch_sequence_output_splits[dev_id], batch_visual_output_splits[dev_id], batch_audio_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu_3_modality, model, parameters_tuple_list, device_ids)
            for key in sim_matrix.keys():
                for idx in range(len(parallel_outputs)):
                    if len(parallel_outputs[idx][key]) > 0:
                        sim_matrix[key] += parallel_outputs[idx][key]
                if len(sim_matrix[key]) > 0:
                    sim_matrix[key] =  np.concatenate(tuple(sim_matrix[key]), axis=0)
              
        else:       
            #sequence_embd = get_embd_from_sequence(model, batch_list_sequence, batch_sequence_output_list)
            sim_matrix={}
             
            
            devc = torch.device('cuda:0')
           
            batch_list_sequence = [tuple(t.to(devc) for t in b) for b in batch_list_sequence]
            
            batch_list_visual = [b[1].to(devc)  for b in batch_list_visual]
            
            batch_list_audio = [b[1].to(devc)  for b in batch_list_audio]
            batch_sequence_output_list=[b.to(devc) for b in batch_sequence_output_list]
            batch_visual_output_list=[b.to(devc) for b in batch_visual_output_list]
            batch_audio_output_list=[b.to(devc) for b in batch_audio_output_list]
           
            sim_matrix_dict = _run_on_single_gpu_3_modality(model, batch_list_sequence, batch_list_visual,batch_list_audio, batch_sequence_output_list, batch_visual_output_list, batch_audio_output_list)
            for key in sim_matrix_dict.keys():
                sim_matrix[key] = np.concatenate(tuple(sim_matrix_dict[key]), axis=0) if  len(sim_matrix_dict[key])>0 else []
           
    logger.info('\t Length-T: {}, Length-V:{}, Length-A:{}'.format(len(sim_matrix['t_va']), len(sim_matrix['t_va'][1]), len(sim_matrix['t_va'][1])))
    
    R1=0
    for key in ['t_a','t_v' ,'t_va']:
        if len(sim_matrix[key])>0:
            metrics = compute_metrics(sim_matrix[key])
            logger.info('\t>>> Retrival method:{}  R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.
                    format(key, metrics['R1'], metrics['R5'], metrics['R10'], metrics['MR']))

            R1 = max(metrics['R1'], R1)
    return R1

DATALOADER_DICT = {'howto100m':dataloader_howto100m, 'audioset':dataloader_audioset,'audioset+howto':dataloader_audioset_howto}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()    
    model = init_model(args, device, n_gpu, args.local_rank)
    only_sim = model.module._stage_one if hasattr(model, 'module') else model._stage_one
    #audio_tokenizer = AudioTokenizer.from_pretrained(args.audio_model, is_mask = only_sim is False)
    if args.do_pretrain:
        assert args.datatype in DATALOADER_DICT.keys()
        train_dataloader, train_length, sampler = DATALOADER_DICT[args.datatype](args, tokenizer, only_sim=only_sim)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        global_step = 0
        epoch = -1
       
        last_optim_state = None
        if args.load_checkpoint:
            epoch, global_step, last_optim_state, model = load_model(epoch, args, n_gpu, device, model, global_step=global_step)
            epoch += 1
            if args.local_rank == 0:
                logger.warning("Will continue to epoch: {}".format(epoch))
        epoch = 0 if epoch < 0 else epoch

        coef_lr = args.coef_lr

        if args.stage_two==True:
             optimizer, scheduler, model = prep_optimizer_clip_s2(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        elif args.stage_two==False:
            optimizer, scheduler, model = prep_optimizer_clip(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
            
        if last_optim_state is not None:
            optimizer.load_state_dict(last_optim_state)

        if args.local_rank == 0:
            logger.info("***** Running pretraining *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        iter_ls_ = [itm for itm in range(args.epochs) if itm >= epoch]
        
        # visdom set
        
        scaler = GradScaler()    
        
        for epoch in iter_ls_:
            sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                            scheduler, global_step, scaler=scaler, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                save_model(epoch, args, model, args.local_rank, type_name="pretrain", global_step=global_step, optimizer=optimizer)
    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch_tva(args, model, test_dataloader, device, n_gpu, meta)

if __name__ == "__main__":
    main()