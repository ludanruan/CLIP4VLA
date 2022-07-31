import h5py
import lmdb
import pickle as pkl
import numpy as np
import torch
import random
SPECIAL_TOKEN_CLIP={"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "<|maskoftext|>", "UNK_TOKEN": "<|unkoftext|>", "PAD_TOKEN": "<|padoftext|>"}

def _compute_mask_indices(
    shape,
    mask_prob: float,
    mask_length: int,
    attention_mask: np.array = None,
    min_masks: int = 0,
) -> np.array:
    """
    Computes random mask spans for a given shape. Used to implement `SpecAugment: A Simple Data Augmentation Method for
    ASR <https://arxiv.org/abs/1904.08779>`__.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )
   
    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length),dtype=np.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = np.ones((batch_size, sequence_length - (mask_length - 1)))

    # get random indices to mask
    spec_aug_mask_idxs = torch.multinomial(torch.tensor(uniform_dist), num_masked_spans).numpy()

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.repeat(spec_aug_mask_idxs[:,:,None],mask_length, \
    axis=-1).reshape(batch_size, num_masked_spans * mask_length)
    
    offsets = np.repeat(np.repeat(np.arange(mask_length)[None, None, :],batch_size, axis=0),\
    num_masked_spans,1).reshape(batch_size, num_masked_spans * mask_length)
    
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    for i in range(batch_size):
        spec_aug_mask[i,spec_aug_mask_idxs[i]] = True
    
   
    if attention_mask is not None:
        # make sure padded input ids cannot be masked
        spec_aug_mask = np.where(attention_mask.bool(), spec_aug_mask, False)

    return spec_aug_mask


class H5Feature(object):
    def __init__(self, h5_paths, feature_framerate=1.0, max_frames=48):
        self.h5_paths = h5_paths
        self.max_frames = max_frames
        if isinstance(h5_paths, str):
            self.h5_paths = [h5_paths]
        self.h5s = []
        self._get_dim()
    
    def get_feature(self, key):
        if self.h5s == []:
            for h5_path in self.h5_paths:
                self.h5s.append(h5py.File(h5_path, 'r'))
        
        fts = []
        for i, h5 in enumerate(self.h5s):
            if key not in h5:
                fts.append(np.zeros((1, self.dims[i]), dtype=np.float32))
                print(f'Feature No.{i} not found. Video name: {key}')
                print('Use zero instead')
            else:
                fts.append(np.array(h5[key], dtype=np.float32))
        try:
            ft = np.concatenate(fts, axis=-1)
        except ValueError:
            fts = self.align_len(fts)
            ft = np.concatenate(fts, axis=-1)
        ft = ft[:self.max_frames]
        return ft
    
    def __getitem__(self, key):
        return self.get_feature(key)
    
    def align_len(self, fts):
        max_len = max([fts[i].shape[0] for i in range(len(fts))])
        for i in range(len(fts)):
            align_index = np.round(np.linspace(0, fts[i].shape[0] - 1, max_len)).astype('int32')
            fts[i] = fts[i][align_index]
        return fts
    
    def _get_dim(self):
        self.dim = 0
        self.dims = []
        for path in self.h5_paths:
            with h5py.File(path) as h5:
                for key, feature in h5.items():
                    self.dim += feature.shape[1]
                    self.dims.append(feature.shape[1])
                    break

class LMDBText(object):
    def __init__(self, lmdb_path, is_pickle=True):
        self.env = lmdb.open(lmdb_path, readonly=True, create=False, readahead=False)
        self.txn = self.env.begin(buffers=True)
        self.is_pickle = is_pickle
    
    def __getitem__(self, key):
        data = self.txn.get(key.encode())
        if self.is_pickle:
            data = pkl.loads(data)
        return data
    
    def __len__(self):
        return self.txn.stat()['entries']

class LMDBFeature(object):
    def __init__(self, lmdb_path, is_pickle=True):
        self.env = lmdb.open(lmdb_path, readonly=True, create=False, readahead=False)
        self.txn = self.env.begin(buffers=True)
        self.is_pickle = is_pickle

    def __getitem__(self, key):

        feature = self.txn.get(key.encode())
        if self.is_pickle:
            feature = pkl.loads(feature)
        return feature
        
    def __len__(self):
        return self.txn.stat()['entries']

class LMDBGroup(object):
    def __init__(self, db_cls, *args, **kwargs):
        self.path2db = {}
        self.db_cls = db_cls
        self.args = args
        self.kwargs = kwargs
    
    def __getitem__(self, path):
        if path in self.path2db:
            return self.path2db[path]
        else:
            db = self.db_cls(path, *self.args, **self.kwargs)
            self.path2db[path] = db
            return db

