from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import calculate_classification_metric
from tqdm import tqdm
import time
from modules.tokenization import SimpleTokenizer as ClipTokenizer
from modules.modeling import  AudioClip
from modules.until_module import AllGather
from util import *
from dataloaders.dataloaders import *
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

allgather = AllGather.apply

torch.distributed.init_process_group(backend="nccl")

global logger
NUM_CLASS_DICT= {'esc50':50, 'audioset':527}
def get_args(description='CLIP4TVA on Classification Task'):
    args = get_parser(description)
    
    
    # Check paramenters
    args.task_type = "classification"
    assert args.datatype in NUM_CLASS_DICT
    args.class_num = NUM_CLASS_DICT[args.datatype]
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
    
    model = AudioClip.from_pretrained( state_dict=model_state_dict, task_config=args)
    
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

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, scaler,local_rank=0):
    global logger
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
        audio:[batchsize,1,max_frames,audio_dim]
        audio_mask:[batchsize,1,max_frames,audio_dim]
        masked_audio:[batchsize,1,max_frames,audio_dim]
        audio_label_index:[batchsize,1,max_frames]????
        '''

        audio, audio_mask, label = batch

        input_dict={'audio':audio, 'audio_mask':audio_mask, 'cls_gt': label}
        
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


def eval_epoch_esc50(args, model, test_dataloader, device, n_gpu, meta=None):
    softmax_func = torch.nn.Softmax(dim=1)
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
   
    with torch.no_grad():
        all_pred = []
        all_target = []
              
        
        for bid, batch in enumerate(test_dataloader):
            
            batch = tuple(t.to(device) for t in batch)

            audio, audio_mask, label = batch
            audio_output, _ = model.get_audio_output(audio, audio_mask, shaped=False)
            input_dict={'audio_output':audio_output, 'audio_mask':audio_mask}
            
            pred = model.classification(**input_dict)
            pred = softmax_func(pred).cpu().detach()

            all_pred.append(pred)
            all_target.append(label.to('cpu').detach())

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

    all_pred = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    metric= calculate_classification_metric(all_pred, all_target)
    logger.info("Classification Acc:{:4f}".format( metric[0]['acc']))
    
    acc = metric[0]['acc']
    return acc

DATALOADER_DICT = {}

DATALOADER_DICT["esc50"] = {"train":dataloader_esc50_classification_train, "val":dataloader_esc50_classification_test}
#DATALOADER_DICT["audioset"] = {"train":dataloader_audioset_classification_train, "val":dataloader_audioset_classification_test}
def run_esc50(args):
    device, n_gpu = init_device(args, args.local_rank)
    tokenizer = ClipTokenizer()
  
    assert args.datatype in DATALOADER_DICT
    model = init_model(args, device, n_gpu, args.local_rank) 
    
    if args.do_train:
        all_fold = "12345"
        acc_dict = {}
        for val in range(1,6):
            val_fold = str(val)
            train_fold = all_fold.replace(val_fold, '')
            test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,val_fold)
            if args.local_rank == 0:
                logger.info("***** Running val on fold {} *****".format(val_fold))
                logger.info("  Num examples = %d", test_length)
                logger.info("  Batch size = %d", args.batch_size_val)
                logger.info("  Num steps = %d", len(test_dataloader))
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer, train_fold)
            num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                            / args.gradient_accumulation_steps) * args.epochs

            coef_lr = args.coef_lr
            
        
            
                
            optimizer, scheduler, model = prep_optimizer_clip_s2(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
            
            if args.local_rank == 0:
                logger.info("***** Running training on fold:{}*****".format(train_fold))
                logger.info("  Num examples = %d", train_length)
                logger.info("  Batch size = %d", args.batch_size)
                logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

            best_acc = 0
            best_output_model_file = None
            global_step = 0
            scaler = GradScaler() 
            for epoch in range(args.epochs):
                train_sampler.set_epoch(epoch)
                tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                scheduler, global_step,scaler, local_rank=args.local_rank)
                if args.local_rank == 0:
                    logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                    output_model_file = save_model(epoch, args, model, type_name="")
                    
                    acc = eval_epoch_esc50(args, model, test_dataloader, device, n_gpu)
                    
                    if best_acc <= acc:
                        best_acc = acc
                        best_output_model_file =  output_model_file
                    logger.info("The best model is: {}, the acc is: {:.4f}".format(best_output_model_file, best_acc))
            acc_dict[val]=best_acc

            model = init_model(args, device, n_gpu, args.local_rank) 
        
        if args.local_rank == 0:

            avg_acc = sum(acc_dict.values())/5
            logger.info("The avg acc is {:.4f}".format(avg_acc))
            # model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            # R1 = eval_epoch_esc50(args, model, test_dataloader, device, n_gpu)
            
                        
    elif args.do_eval:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
        if args.local_rank == 0:
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", test_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(test_dataloader))
        if args.local_rank == 0:
            
            eval_epoch_esc50(args, model, test_dataloader, device, n_gpu)

def main():
    global logger
    
    args = get_args()
    args = set_seed_logger(args)
    
    args.freeze='audio'
    run_esc50(args)
    
if __name__ == "__main__":
    main()