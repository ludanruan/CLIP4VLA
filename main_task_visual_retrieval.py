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
from metrics import t2v_metrics as compute_metrics
from metrics import sim_rerank
from tqdm import tqdm
import time
import argparse
from modules.tokenization import SimpleTokenizer as ClipTokenizer
from modules.tokenization import END_TOKEN
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import AudioClip
from modules.optimization import BertAdam
from dataloaders.dataloaders import *
from util import *
from utils.visualizer import Visualizer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from modules.until_module import AllGather
allgather = AllGather.apply
torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='UniVL on Retrieval Task'):
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
        model_state_dict = torch.load(args.init_model, map_location='cpu')
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
        model = AudioClip.from_pretrained(args.audio_model,args.cross_model, args.decoder_model, 
                                cache_dir=args.cache_dir, state_dict=model_state_dict, task_config=args)
        
        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step,scaler, local_rank=0):
    
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
   
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        '''
        input_ids:[batchsize,1,max_words]
        input_mask:[batchsize,1,max_words]
        segment_ids:[batchsize,1,] input text token type
        vdieo:[batchsize,1,max_frames,video_dim]
        video_mask:[batchsize,1,max_frames,video_dim]
        masked_video:[batchsize,1,max_frames,video_dim]
        video_label_index:[batchsize,1,max_frames]????
        '''
        
        
        input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = batch[0:9]

        input_dict={'input_ids':input_ids, 'attention_mask':input_mask, \
                        'video':video, 'video_mask':video_mask,
                        'pairs_masked_text':pairs_masked_text,'pairs_token_labels':pairs_token_labels,
                        'masked_video':masked_video, 'video_labels_index':video_labels_index}
        if args.refine_nce:
            video_idx = batch[-1]
            input_dict.update({'video_idx':video_idx})
        
        if args.fp16:
            with autocast():
                loss, loss_recoder = model(**input_dict)
        else:
            loss, loss_recoder = model(**input_dict)
        
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        

        if args.fp16:   
            scaler.scale(loss).backward() 
           
        else:
            loss.backward()
               

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
                
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

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

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def _run_on_single_gpu(model, batch_list_t, batch_list_modal2, batch_sequence_output_list, batch_modal2_output_list , modal1='text', modal2='video', hard_negative=False):
    sim_matrix = []
    
    for idx1, b1 in tqdm(enumerate(batch_list_t)):
        input_ids, input_mask, segment_ids = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_modal2):
            modal2_mask  = b2
            modal2_output = batch_modal2_output_list[idx2]
            b1b2_logits = model.get_similarity_logits(sequence_output, modal2_output, input_mask, modal2_mask, modal1, modal2, input_ids=input_ids, hard_negative=hard_negative)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
        #batchsize,text_len 
    return sim_matrix

def get_embd_from_sequence(model, batch_list_t, batch_sequence_output_list):
    all_embd = []
    
    for idx1, b1 in enumerate(batch_list_t):
        input_ids, input_mask, segment_ids = b1
        sequence_output = batch_sequence_output_list[idx1]
        batch_embd = model._mean_pooling_for_single_modal(sequence_output, input_mask.squeeze(1), "text")
        all_embd.append(batch_embd)
    
    all_embd = torch.cat(all_embd, axis=0)
    return all_embd

def save_visual_web(args, sim, metrics, meta, weights=None, choose_index=None):
    
    name = '_'.join([args.datatype, "visual_retrieval"])
    visualizer = Visualizer(name= name,  web_dirs=args.web_dirs)
    query_masks = np.ones((sim.shape[0],1), dtype=int)
    nested_metrics = metrics
    localtime = time.asctime(time.localtime())
    subdir_names = args.output_dir.split('/')
    subdir_name = "{}/{}/R@1_{:.2f}_{}".format(subdir_names[-2], subdir_names[-1], metrics["R1"], localtime)

    
    modalities = ['clip']   
         
    
    visualizer.visualize_ranking(
              args=args,
              sims=sim,
              query_masks=query_masks,
              meta=meta,
              nested_metrics=nested_metrics,
              modalities=modalities,
              subdir_name=subdir_name,
              choose_index=choose_index
          )

    return

def pick_index(sim_matrix):
    
    sim_matrix_ta = sim_matrix['t_a']
    sim_matrix_tv = sim_matrix['t_v']
    sim_matrix_tav = sim_matrix['t_va']
    sorted_dists_ta = np.sort(-sim_matrix_ta, axis=1)[:,0]
    gt_ta = np.diagonal(-sim_matrix_ta) 

    sorted_dists_tv = np.sort(-sim_matrix_tv, axis=1)[:,0]
    gt_tv = np.diagonal(-sim_matrix_tv) 

    sorted_dists_tav = np.sort(-sim_matrix_tav, axis=1)[:,0]
    gt_tav = np.diagonal(-sim_matrix_tav) 

    candidates = np.where((gt_tv - sorted_dists_tv!=0 ) & (gt_tav - sorted_dists_tav == 0))[0]
    return candidates



def eval_epoch_tv(args, model, test_dataloader, device, n_gpu, meta=None):
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
   
    with torch.no_grad():
        batch_list_sequence = []
        batch_list_visual = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        
       
        for bid, batch in enumerate(test_dataloader):
            
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, video, video_mask= batch[0:5] 
           
            sequence_output = model.get_sequence_output(input_ids,  input_mask).cpu().detach()
            
            if args.train_sim_after_cross is False:
                sequence_output = sequence_output[torch.arange(input_ids.shape[0]), (input_ids == END_TOKEN).nonzero(as_tuple=True)[-1]]
            
            batch_sequence_output_list.append(sequence_output)
            batch_list_sequence.append([input_ids.cpu().detach(), input_mask.cpu().detach(), segment_ids.cpu().detach()])
            
            loader_start_, loader_end_ = bid * args.batch_size_val,  bid * args.batch_size_val + video.size(0)
            modal_filtered = [i% args.multi_sentence==0 for i in range(loader_start_, loader_end_)]
            
            if sum(modal_filtered) > 0:
                video = video[modal_filtered].to(torch.float32)
               
                video_mask = video_mask[modal_filtered].to(torch.float32)

                visual_output = model.get_visual_output(video, video_mask, shaped=False).cpu().detach()
                
                batch_visual_output_list.append(visual_output)
                
                batch_list_visual.append([video.cpu().detach(), video_mask.cpu().detach()])

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_sequence_splits = []
            batch_list_visual_splits = []
            batch_sequence_output_splits = []
            batch_visual_output_splits = []
            batch_len = len(batch_list_sequence)
            split_len = (batch_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                      
                devc = torch.device('cuda:{}'.format(str(dev_id)))
                devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_sequence[s_:e_]]
                batch_list_sequence_splits.append(devc_batch_list)
                devc_batch_list = [b[1].to(devc) for b in batch_list_visual]
                batch_list_visual_splits.append(devc_batch_list)

                devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                batch_sequence_output_splits.append(devc_batch_list)
                devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                batch_visual_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_sequence_splits[dev_id], batch_list_visual_splits[dev_id],
                                      batch_sequence_output_splits[dev_id], batch_visual_output_splits[dev_id], 'text', 'video') for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

              
        else:
            devc = torch.device('cuda:0')
           
            batch_list_sequence = [tuple(t.to(devc) for t in b) for b in batch_list_sequence]
            
            batch_list_visual = [b[1].to(devc)  for b in batch_list_visual]          
            batch_sequence_output_list=[b.to(devc) for b in batch_sequence_output_list]
            batch_visual_output_list=[b.to(devc) for b in batch_visual_output_list]
            
            sim_matrix = _run_on_single_gpu(model, batch_list_sequence, batch_list_visual, batch_sequence_output_list, batch_visual_output_list, 'text', 'video')
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            
            
           

    print('sim_matrix:{}'.format(sim_matrix))
    metrics = compute_metrics(sim_matrix)
    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    logger.info('sim_matrix\t>>>  R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.
                format(metrics['R1'], metrics['R5'], metrics['R10'], metrics['MR']))

    if args.do_visualize and meta is not None:
        save_visual_web(args, sim_matrix, metrics, meta )

    R1 = metrics['R1']
    return R1

DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_retrieval_train, "val":dataloader_youcook_retrieval_test}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_retrieval_train, "val":dataloader_msrvtt_retrieval_test}
DATALOADER_DICT["audiocaps"] = {"train":dataloader_audiocaps_retrieval_train, "val":dataloader_audiocaps_retrieval_test}
DATALOADER_DICT["activitynet"] = {"train":dataloader_activitynet_retrieval_train, "val":dataloader_activitynet_retrieval_test}
DATALOADER_DICT["vatex"] = {"train":dataloader_vatex_retrieval_train, "val":dataloader_vatex_retrieval_test}
def main():
    global logger
    
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
   
    tokenizer = ClipTokenizer()
        
    args.transform_train=None
    args.transform_test=None

    assert args.datatype in DATALOADER_DICT
    model = init_model(args, device, n_gpu, args.local_rank)

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length, meta = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        
        if args.train_sim_after_cross:
            optimizer, scheduler, model = prep_optimizer_clip_s2(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        else:
            optimizer, scheduler, model = prep_optimizer_clip(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        scaler = GradScaler() 
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, scaler,local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                
                output_model_file = save_model(epoch, args, model, type_name="")
                
                R1 = eval_epoch_tv(args, model, test_dataloader, device, n_gpu)
                
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        if args.local_rank == 0:
           
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            
            R1 = eval_epoch_tv(args, model, test_dataloader, device, n_gpu)
            
                  
    elif args.do_eval:
        if args.local_rank == 0:
            
            eval_epoch_tv(args, model, test_dataloader, device, n_gpu, meta)
                

if __name__ == "__main__":
    main()