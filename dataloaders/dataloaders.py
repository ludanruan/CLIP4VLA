import torch, pdb
import pickle
import logging
from torch.utils.data import (SequentialSampler)
from torch.utils.data import DataLoader
from dataloaders.data import LMDBGroup, LMDBFeature
from dataloaders.dataloader_howto100m import Youtube_DataLoader
from dataloaders.dataloader_audioset import Audioset_DataLoader
from dataloaders.dataloader_audioset_howto import Audioset_Howto_DataLoader
from dataloaders.dataloader_clotho_caption import Clotho_Caption_DataLoader
from dataloaders.dataloader_audiocaps_retrieval import Audiocaps_Retrieval_DataLoader
from dataloaders.dataloader_audiocaps_caption import Audiocaps_Caption_DataLoader
from dataloaders.dataloader_vatex_retrieval import Vatex_Retrieval_DataLoader
from dataloaders.dataloader_vatex_caption import Vatex_Caption_DataLoader
from dataloaders.dataloader_youcook_retrieval import Youcook_Retrieval_DataLoader
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_Retrieval_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_Retrieval_TrainDataLoader
from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
from dataloaders.dataloader_activitynet_caption import Activitynet_Caption_DataLoader
from dataloaders.dataloader_activitynet_retrieval import Activitynet_Retrieval_DataLoader
from dataloaders.dataloader_esc50_classification import Esc50_Classification_DataLoader
logger = logging.getLogger(__name__)

def dataloader_howto100m_test(args, tokenizer, only_sim=True):
    if args.local_rank == 0:
        logger.info('Loading captions: {}'.format(args.data_path))
    data_dict = pickle.load(open(args.data_path, 'rb'))
    
    dataset = Youtube_DataLoader(
        csv=args.val_csv,
        features_path=args.features_path,
        data_dict=data_dict,
        min_time=5.0,
        max_words=args.max_words,
        min_words=0,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=1,
        max_frames=args.max_frames,
        use_mil=False,
        only_sim=only_sim,
        sampled_use_mil=False,
        pretrain_enhance_vmodal=False,
        video_dim=args.video_dim,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
     
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen
    )

    if args.local_rank == 0:
        logger.info('Done, data_dict length: {}'.format(len(dataset)))
        
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    
    return dataloader, len(dataset), None


def dataloader_howto100m(args, tokenizer, only_sim=False):
    if args.local_rank == 0:
        logger.info('Loading captions: {}'.format(args.data_path))
    data_dict = pickle.load(open(args.data_path, 'rb'))
    
    use_lmdb = False if not hasattr(args, 'use_lmdb') else args.use_lmdb
    print(f'Use lmdb: {use_lmdb}')
    if use_lmdb:
     
        lmdb_group_howto = LMDBGroup(db_cls=LMDBFeature, is_pickle=True)
        feature_db = lmdb_group[args.features_path]
    else:
        feature_db = None

    dataset = Youtube_DataLoader(
        csv = args.train_csv,
        features_path = args.features_path,
        data_dict = data_dict,
        min_time = args.min_time,
        max_words = args.max_words,
        min_words = args.min_words,
        feature_framerate = args.feature_framerate,
        tokenizer = tokenizer,
        n_pair = args.n_pair,
        max_frames = args.max_frames,
        use_mil = args.use_mil,
        only_sim = only_sim,
        sampled_use_mil = args.sampled_use_mil,
        pretrain_enhance_vmodal = args.pretrain_enhance_vmodal,
        video_dim = args.video_dim,
        frame_order = args.train_frame_order,
        slice_framepos = args.slice_framepos,
        
        use_lmdb = use_lmdb,
        feat_db = feature_db,

        audio_path = args.audio_path,
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        with_decoder = args.with_decoder,
        enhance_single_modal = args.enhance_single_modal
    )
    if args.local_rank == 0:
        logger.info('Done, data_dict length: {}'.format(len(data_dict)))
    #sampler = SequentialSampler(dataset)    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    
    return dataloader, len(dataset), sampler

def dataloader_audioset(args, tokenizer, only_sim=False):
    if args.local_rank == 0:
        logger.info('Loading captions: {}'.format(args.data_path))
    data_dict = pickle.load(open(args.data_path, 'rb'))
    
    use_lmdb = False if not hasattr(args, 'use_lmdb') else args.use_lmdb
    print(f'Use lmdb: {use_lmdb}')
    if use_lmdb:
        lmdb_group = LMDBGroup(db_cls=LMDBFeature, is_pickle=True)
        feature_db = lmdb_group[args.features_path]
    else:
        feature_db = None

    dataset = Audioset_DataLoader(
        csv=args.train_csv,
        features_path=args.features_path,
        data_dict=data_dict,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=args.n_pair,
        max_frames=args.max_frames,
        use_mil=args.use_mil,
        only_sim=only_sim,
        sampled_use_mil=args.sampled_use_mil,
        pretrain_enhance_vmodal=args.pretrain_enhance_vmodal,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        
        use_lmdb=use_lmdb,
        feat_db=feature_db,

        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        with_decoder = args.with_decoder,
        enhance_single_modal= args.enhance_single_modal
    )
    if args.local_rank == 0:
        logger.info('Done, data_dict length: {}'.format(len(data_dict)))
    #sampler = SequentialSampler(dataset)    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    
    return dataloader, len(dataset), sampler

def dataloader_audioset_howto(args, tokenizer, only_sim=False):
    if args.local_rank == 0:
        logger.info('Loading captions: {}'.format(args.data_path))
    data_path_audioset, data_path_howto = args.data_path.split(',')
    assert 'audioset' in data_path_audioset and 'howto100m' in data_path_howto, print('Please check data_path carefully')
    data_dict_howto= pickle.load(open(data_path_howto, 'rb'))
    data_dict_audioset= pickle.load(open(data_path_audioset, 'rb'))
    
    data_dict={'howto100m':data_dict_howto, 'audioset':data_dict_audioset}
  
    features_path_audioset, features_path_howto= args.features_path.split(',')
    #assert 'audioset' in features_path_audioset and 'howto100m' in features_path_howto, print('Please check feature_paths carefully')
    features_path = {'howto100m':features_path_howto, 'audioset':features_path_audioset}
    
    train_csv_audioset, train_csv_howto = args.train_csv.split(',')
    assert 'audioset' in train_csv_audioset and 'howto100m' in train_csv_howto, print('Please check train_csv carefully')
    train_csv = {'howto100m':train_csv_howto, 'audioset':train_csv_audioset}

    audio_path_audioset, audio_path_howto = args.audio_path.split(',')
    #assert 'audioset' in audio_path_audioset and 'howto100m' in audio_path_howto, print('Please check audio_path carefully')
    audio_path = {'howto100m':audio_path_howto, 'audioset':audio_path_audioset}
  
    use_lmdb = False if not hasattr(args, 'use_lmdb') else args.use_lmdb
    print(f'Use lmdb: {use_lmdb}')
   
    if use_lmdb:
        
        lmdb_video_howto = LMDBGroup(db_cls=LMDBFeature, is_pickle=True)
        feature_db_howto = lmdb_video_howto[features_path['howto100m']]
        lmdb_video_audioset = LMDBGroup(db_cls=LMDBFeature, is_pickle=True)
        feature_db_audioset = lmdb_video_audioset[features_path['audioset']]
        feature_db_video = {'howto100m':feature_db_howto, 'audioset':feature_db_audioset}

        lmdb_audio_howto = LMDBGroup(db_cls=LMDBFeature, is_pickle=True)
        audio_db_howto = lmdb_audio_howto[audio_path['howto100m']]
        lmdb_audio_audioset = LMDBGroup(db_cls=LMDBFeature, is_pickle=True)
        audio_db_audioset = lmdb_audio_audioset[audio_path['audioset']]
        feature_db_audio = {'howto100m':audio_db_howto, 'audioset':audio_db_audioset}
        feature_db = {'video':feature_db_video, 'audio':feature_db_audio}
    else:
        feature_db = None

    dataset = Audioset_Howto_DataLoader(
        csv=train_csv,
        features_path=features_path,
        data_dict=data_dict,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=args.n_pair,
        max_frames=args.max_frames,
        use_mil=args.use_mil,
        only_sim=only_sim,
        sampled_use_mil=args.sampled_use_mil,
        pretrain_enhance_vmodal=args.pretrain_enhance_vmodal,
        video_dim=args.video_dim,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,

        use_lmdb = use_lmdb,
        feat_db = feature_db,

        audio_path=audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        with_decoder = args.with_decoder,
        enhance_single_modal = args.enhance_single_modal
    )
    if args.local_rank == 0:
        logger.info('Done, data_dict length: {}'.format(len(data_dict['audioset'])+len(data_dict['howto100m'])))
    #sampler = SequentialSampler(dataset)    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    
    return dataloader, len(dataset), sampler


def dataloader_clotho_caption_train(args, tokenizer):
    
    clotho_dataset = Clotho_Caption_DataLoader(
        csv=args.train_csv,
        caption_path=args.data_path,
        meta_path=args.meta_path,
        max_words=args.max_words, 
        max_frames=args.max_frames, 
        tokenizer=tokenizer,
     
        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
  
        video_path=args.raw_video_path,
        with_decoder=args.with_decoder,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(clotho_dataset)
    dataloader = DataLoader(
        clotho_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(clotho_dataset), train_sampler

def dataloader_clotho_caption_test(args, tokenizer):
    clotho_testset = Clotho_Caption_DataLoader(
        csv=args.val_csv,
        caption_path=args.data_path,
        meta_path=args.meta_path,
        max_words=args.max_words,
        max_frames=args.max_frames,
        tokenizer=tokenizer,
        
        
        audio_path=args.audio_path,
      
        
        max_audio_length = args.max_audio_length,
 
        video_path=args.raw_video_path,
        with_decoder = args.with_decoder,
    )
    meta = clotho_testset.get_meta()
    test_sampler = SequentialSampler(clotho_testset)
    dataloader_clotho = DataLoader(
        clotho_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )
    logger.info('Clotho validation pairs: {}'.format(len(clotho_testset)))

    return dataloader_clotho, len(clotho_testset), meta

def dataloader_youcook_caption_train(args, tokenizer):
    youcook_dataset = Youcook_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,

        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        video_path=args.raw_video_path,
        filter_video_id = args.filter_video_id
        
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_caption_test(args, tokenizer):
    youcook_testset = Youcook_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,

        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        video_path=args.raw_video_path,
        filter_video_id = args.filter_video_id
    )

    meta = youcook_testset.get_meta()
    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset), meta

def dataloader_msrvtt_caption_test(args, tokenizer, split_type="test"):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        split_type=split_type,

        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = msrvtt_testset.get_meta()

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset), meta

def dataloader_msrvtt_caption_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        split_type="train",

        audio_path=args.audio_path,
       
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        filter_video_id = args.filter_video_id
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_activitynet_caption_train(args, tokenizer):
    
    activitynet_dataset = Activitynet_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        video_path = args.raw_video_path,
        
        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        extend = args.extend,
        filter_video_id = args.filter_video_id
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activitynet_dataset)
    dataloader = DataLoader(
        activitynet_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activitynet_dataset), train_sampler

def dataloader_activitynet_caption_test(args, tokenizer):

    activitynet_testset = Activitynet_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,      

        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        video_path = args.raw_video_path,
        extend = args.extend,
        filter_video_id = args.filter_video_id
    )

    meta = activitynet_testset.get_meta()
    test_sampler = SequentialSampler(activitynet_testset)
    dataloader_activitynet = DataLoader(
        activitynet_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('Activitynet validation pairs: {}'.format(len(activitynet_testset)))
    return dataloader_activitynet, len(activitynet_testset), meta

def dataloader_activitynet_retrieval_train(args, tokenizer):
    
    activitynet_dataset = Activitynet_Retrieval_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,


        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        video_path=args.raw_video_path,
        filter_video_id = args.filter_video_id
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activitynet_dataset)
    dataloader = DataLoader(
        activitynet_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True
    )

    return dataloader, len(activitynet_dataset), train_sampler

def dataloader_activitynet_retrieval_test(args, tokenizer):
    activitynet_testset = Activitynet_Retrieval_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        
        audio_path=args.audio_path,
        max_audio_length = args.max_audio_length,
        audio_tokenlen = args.audio_tokenlen,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        video_path=args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = activitynet_testset.get_meta()
    #test_sampler = torch.utils.data.distributed.DistributedSampler(activitynet_testset)
    test_sampler = SequentialSampler(activitynet_testset)
    dataloader_activitynet = DataLoader(
        activitynet_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False
    )
    if args.rank==0:
        logger.info('Activitynet validation pairs: {}'.format(len(activitynet_testset)))

    return dataloader_activitynet, len(activitynet_testset), meta


def dataloader_youcook_retrieval_train(args, tokenizer):
    
    youcook_dataset = Youcook_Retrieval_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        slice_framepos=args.slice_framepos,

        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
        video_path=args.raw_video_path,
        filter_video_id = args.filter_video_id
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_retrieval_test(args, tokenizer):
    youcook_testset = Youcook_Retrieval_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        slice_framepos=args.slice_framepos,
        
        
        audio_path=args.audio_path,
        
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
        video_path=args.raw_video_path,
        filter_video_id = args.filter_video_id

    )
    meta = youcook_testset.get_meta()
    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )
    logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))

    return dataloader_youcook, len(youcook_testset), meta

def dataloader_msrvtt_retrieval_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Retrieval_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        unfold_sentences=args.expand_msrvtt_sentences,
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,  
        filter_video_id = args.filter_video_id    
        
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_retrieval_test(args, tokenizer):
    msrvtt_testset = MSRVTT_Retrieval_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
       
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = msrvtt_testset.get_meta()
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset), meta

def dataloader_audiocaps_retrieval_test(args, tokenizer):
    audiocaps_testset = Audiocaps_Retrieval_DataLoader(
        csv=args.val_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
 
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = audiocaps_testset.get_meta()
    dataloader_audiocaps = DataLoader(
        audiocaps_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_audiocaps, len(audiocaps_testset), meta

def dataloader_audiocaps_retrieval_train(args, tokenizer):
    audiocaps_trainset = Audiocaps_Retrieval_DataLoader(
        csv=args.train_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,

        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(audiocaps_trainset)
    dataloader = DataLoader(
        audiocaps_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return  dataloader, len(audiocaps_trainset), train_sampler

def dataloader_audiocaps_caption_train(args, tokenizer):
    
    audiocaps_dataset = Audiocaps_Caption_DataLoader(
        csv=args.train_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words, 
        max_frames=args.max_frames, 
        tokenizer=tokenizer,
        feature_framerate=args.feature_framerate,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
        
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(audiocaps_dataset)
    dataloader = DataLoader(
        audiocaps_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(audiocaps_dataset), train_sampler

def dataloader_audiocaps_caption_test(args, tokenizer):
    audiocaps_testset = Audiocaps_Caption_DataLoader(
        csv=args.val_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
       
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = audiocaps_testset.get_meta()
    test_sampler = SequentialSampler(audiocaps_testset)
    dataloader_audiocaps = DataLoader(
        audiocaps_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )
    logger.info('Audiocaps validation pairs: {}'.format(len(audiocaps_testset)))

    return dataloader_audiocaps, len(audiocaps_testset), meta


def dataloader_vatex_retrieval_test(args, tokenizer):
    vatex_testset = Vatex_Retrieval_DataLoader(
        csv=args.val_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
 
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = vatex_testset.get_meta()
    dataloader_vatex = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_vatex, len(vatex_testset), meta

def dataloader_vatex_retrieval_train(args, tokenizer):
    vatex_trainset = Vatex_Retrieval_DataLoader(
        csv=args.train_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,

        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_trainset)
    dataloader = DataLoader(
        vatex_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return  dataloader, len(vatex_trainset), train_sampler

def dataloader_vatex_caption_train(args, tokenizer):
    
    vatex_dataset = Vatex_Caption_DataLoader(
        csv=args.train_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words, 
        max_frames=args.max_frames, 
        tokenizer=tokenizer,
        feature_framerate=args.feature_framerate,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
        
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
    dataloader = DataLoader(
        vatex_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vatex_dataset), train_sampler

def dataloader_vatex_caption_test(args, tokenizer):
    vatex_testset = Vatex_Caption_DataLoader(
        csv=args.val_csv,
        caption_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,

        
        audio_path=args.audio_path,
        
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        audio_tokenlen = args.audio_tokenlen,
       
        video_path = args.raw_video_path,
        filter_video_id = args.filter_video_id
    )
    meta = vatex_testset.get_meta()
    test_sampler = SequentialSampler(vatex_testset)
    dataloader_vatex = DataLoader(
        vatex_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )
    logger.info('vatex validation pairs: {}'.format(len(vatex_testset)))

    return dataloader_vatex, len(vatex_testset), meta

def dataloader_esc50_classification_test(args, tokenizer, split='1'):
    esc50_testset = Esc50_Classification_DataLoader(
        csv_path=args.val_csv,
        max_words=args.max_words,
        tokenizer=tokenizer,

        audio_path=args.audio_path,
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,

        split = split
    )
    
    test_sampler = SequentialSampler(esc50_testset)
    dataloader_esc50 = DataLoader(
        esc50_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False
    )
    if args.rank==0:
        logger.info('Esc50 validation pairs: {}'.format(len(esc50_testset)))

    return dataloader_esc50, len(esc50_testset)

def dataloader_esc50_classification_train(args, tokenizer, split='2345'):
    esc50_trainset = Esc50_Classification_DataLoader(
        csv_path=args.train_csv,
        max_words=args.max_words,
        tokenizer=tokenizer,
        audio_path=args.audio_path,
        max_audio_length = args.max_audio_length,
        audio_rate = args.audio_rate,
        audio_channel = args.audio_channel,
        split = split
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(esc50_trainset)
    dataloader_esc50 = DataLoader(
        esc50_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    if args.rank==0:
        logger.info('Esc50 training pairs: {}'.format(len(esc50_trainset)))

    return dataloader_esc50, len(esc50_trainset), train_sampler
