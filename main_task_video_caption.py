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
from collections import OrderedDict
from nlgeval import NLGEval
from tqdm import tqdm
import time
import argparse
from modules.tokenization import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import AudioClip
from modules.optimization import BertAdam
from modules.beam import Beam
from util import *
from dataloaders.dataloaders import *
from utils.visualizer import Visualizer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from dataloaders.data import SPECIAL_TOKEN_CLIP
torch.distributed.init_process_group(backend="nccl")

def get_args(description='CLIP4VLA on Caption Task'):
    args = get_parser(description)

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
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
       
        model = AudioClip.from_pretrained(state_dict=model_state_dict, task_config=args)
        
        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, scaler,nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
       
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
         
        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
        audio, audio_mask, masked_audio, audio_labels_index = batch
        input_dict = {'input_ids':input_ids,  'attention_mask':input_mask, \
                    'video':video, 'video_mask':video_mask, 
                    'pairs_masked_text':pairs_masked_text, 'pairs_token_labels':pairs_token_labels,
                    'masked_video':masked_video, 'video_labels_index':video_labels_index,
                    'audio':audio, 'audio_mask':audio_mask, 'masked_audio':masked_audio, \
                    'audio_labels_index':audio_labels_index, 'input_caption_ids':pairs_input_caption_ids, \
                    'decoder_mask':pairs_decoder_mask, 'output_caption_ids':pairs_output_caption_ids}
            
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
            #loss.backward()   
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


# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)

    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)
    
    if len(input_tuples) == 5:
        sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt = input_tuples
   
    else:
        sequence_output_rpt, visual_output_rpt, audio_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, audio_mask_rpt = input_tuples
        active_audio_output_rpt = collect_active_part(audio_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
        active_audio_mask_rpt = collect_active_part(audio_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
   

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
    if len(input_tuples) == 5:
        return (active_sequence_output_rpt, active_visual_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt), \
           active_inst_idx_to_position_map
    else:
        return (active_sequence_output_rpt, active_visual_output_rpt,active_audio_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt, active_audio_mask_rpt), \
           active_inst_idx_to_position_map


def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
       
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
    
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)
        
        sequence_output_rpt, visual_output_rpt, audio_output_rpt, \
        input_ids_rpt, input_mask_rpt, video_mask_rpt, audio_mask_rpt = input_tuples
        dec_output = decoder(sequence_output_rpt, visual_output_rpt, audio_output_rpt, input_ids_rpt, input_mask_rpt,
                             video_mask_rpt, audio_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)
 
    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def save_visual_web(args, results, metrics, meta):
    
    name = '_'.join([args.datatype, args.task_type])
    visualizer = Visualizer(name= name,  web_dirs=args.web_dirs)
   
    
    localtime = time.asctime(time.localtime())
    subdir_names = args.output_dir.split('/')
    subdir_name = "{}/{}/Cider_{:.2f}_{}".format(subdir_names[-2], subdir_names[-1], metrics["CIDEr"], localtime)

    
    modalities= 'vision+audio'  
    
    visualizer.visualize_caption(
              args=args,  
              results=results,
              meta=meta,
              metrics=metrics,
              modalities=modalities,
              subdir_name=subdir_name,
              
          )

    return

def eval_epoch_tva(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None, meta=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._stage_one:
        return 0.

    all_result_lists = []
    all_caption_lists = []
    model.eval()
    
    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
     
        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
        audio, audio_mask, masked_audio, audio_labels_index = batch

        with torch.no_grad():
            
            # input ids is None:[[start], [mask]]
            sequence_output= model.get_sequence_output(input_ids, input_mask)
            visual_output = model.get_visual_output(video, video_mask)
            audio_output,_ = model.get_audio_output(audio, audio_mask)

            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = sequence_output.device
            n_inst, len_s, d_h = sequence_output.size()
            _, len_v, v_h = visual_output.size()
            _, len_a, a_h = audio_output.size()

            
            
            decoder= model.cross_caption
            

            # Note: shaped first, then decoder need the parameter shaped=True
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            input_mask = input_mask.view(-1, input_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])


            sequence_output_rpt = sequence_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            audio_output_rpt = audio_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_a, a_h)
            input_ids_rpt = input_ids.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            input_mask_rpt = input_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            audio_mask_rpt = audio_mask.repeat(1, n_bm).view(n_inst * n_bm, len_a)
          
            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, special_token_dict=args.special_token_dict, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
         
            for len_dec_seq in range(1, args.max_words + 1):
             
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (sequence_output_rpt, visual_output_rpt, audio_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, audio_mask_rpt))

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (sequence_output_rpt, visual_output_rpt, audio_output_rpt,  input_ids_rpt, input_mask_rpt, video_mask_rpt, audio_mask_rpt), \
                inst_idx_to_position_map = collate_active_info((sequence_output_rpt, visual_output_rpt, audio_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, audio_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if args.special_token_dict["SEP_TOKEN"] in decode_text_list:
                    SEP_index = decode_text_list.index(args.special_token_dict["SEP_TOKEN"])
                    decode_text_list = decode_text_list[:SEP_index]
                if args.special_token_dict["PAD_TOKEN"] in decode_text_list:
                    PAD_index = decode_text_list.index(args.special_token_dict["PAD_TOKEN"])
                    decode_text_list = decode_text_list[:PAD_index]
                # if decode_text_list[-1] != '.':
                #     decode_text_list.append('.')
                decode_text = ' '.join(decode_text_list)
                
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if args.special_token_dict["SEP_TOKEN"] in decode_text_list:
                    SEP_index = decode_text_list.index(args.special_token_dict["SEP_TOKEN"])
                    decode_text_list = decode_text_list[:SEP_index]
                if args.special_token_dict["PAD_TOKEN"] in decode_text_list:
                    PAD_index = decode_text_list.index(args.special_token_dict["PAD_TOKEN"])
                    decode_text_list = decode_text_list[:PAD_index]
                # if decode_text_list[-1] != '.':
                #     decode_text_list.append('.')
                decode_text = ' '.join(decode_text_list)
                
                all_caption_lists.append(decode_text)

    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.data_dict[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            
            writer.write(ground_txt + "\n")
    

    if args.datatype in ["msrvtt" ,"audiocaps", "vatex"]:
        
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    # Evaluate
    metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)
    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))
    
    if args.do_visualize and args.do_eval and meta is not None:
        save_visual_web(args, all_result_lists, metrics_nlg, meta)
    
    Cider = metrics_nlg["CIDEr"]
    return Cider


DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_caption_train, "val":dataloader_youcook_caption_test}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_caption_train, "val":dataloader_msrvtt_caption_test}
DATALOADER_DICT["activitynet"] = {"train":dataloader_activitynet_caption_train, "val":dataloader_activitynet_caption_test}
DATALOADER_DICT["audiocaps"] = {"train":dataloader_audiocaps_caption_train, "val":dataloader_audiocaps_caption_test}
DATALOADER_DICT["vatex"] = {"train":dataloader_vatex_caption_train, "val":dataloader_vatex_caption_test}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()
    assert args.task_type == "caption"
    model = init_model(args, device, n_gpu, args.local_rank)
    
    args.special_token_dict = SPECIAL_TOKEN_CLIP
    skip_eval = SKIP_STEPS[args.datatype]
    
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

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
        
        optimizer, scheduler, model =  prep_optimizer_clip_s2(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        
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

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, scaler=scaler,nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, type_name="")
                if epoch > skip_eval:
                    Cider = eval_epoch_tva(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    if best_score <= Cider:
                        best_score = Cider
                        best_output_model_file = output_model_file
                    logger.info("The best model is: {}, the Cider is: {:.4f}".format(best_output_model_file, best_score))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch_tva(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj, meta=meta)
    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch_tva(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj, meta=meta)

if __name__ == "__main__":
    main()