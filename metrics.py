from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import editdistance

def t2v_metrics(sims, query_masks=None, query_masks_class_t2v=None,
                query_masks_class_v2t=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"

    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    # if False:
    #     import sys
    #     import matplotlib
    #     from pathlib import Path
    #     matplotlib.use("Agg")
    #     import matplotlib.pyplot as plt
    #     sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
    #     from zsvision.zs_iterm import zs_dispFig # NOQA
    #     plt.matshow(dists)
    #     zs_dispFig()
    #     import ipdb; ipdb.set_trace()

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_vids))
              for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)]
              for jj in range(num_vids)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin, 
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    # break_ties = "optimistically"
    break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    

    if query_masks is not None:
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()

    return cols2metrics(cols, num_queries, query_masks_class_t2v)



def calculate_classification_metric(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """
    
    batch_size, classes_num = output.shape
    if target.dim() == 1:
        target =  F.one_hot(target,classes_num)

    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
 
    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        if target[:, k].sum()==0:continue
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)
   
    return stats



def cols2metrics(cols, num_queries, query_masks_class=None):
    if query_masks_class is not None:
        new_query_number = int(num_queries) - np.count_nonzero(query_masks_class==0)
        new_cols = np.zeros(new_query_number)
        counter = 0
        query_masks_class = query_masks_class.reshape(cols.shape)
        for loc, query_mask in enumerate(query_masks_class):
            if query_mask == 1:
                new_cols[counter] = cols[loc]
                counter += 1
        cols = new_cols
        num_queries = new_query_number
    metrics = {}
    metrics["R1"] =  float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = float(np.sum(cols < 5)) / num_queries
    metrics["R10"] =  float(np.sum(cols < 10)) / num_queries
    metrics["MR"] = np.median(cols) + 1
    #metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
   
    return metrics


def compute_metrics_many2one(x, many=5):
    
    assert x.shape[0] % many == 0
   
    x = x[:,::many] 
    gt = np.zeros_like(x)
    index_x = range(x.shape[0])
    index_y = [i_x // many for i_x in index_x]
    gt[index_x,index_y] = 1
    

    sx = np.sort(-x, axis=1)
    
    d =-x[gt==1]
    d = d[:, np.newaxis]
    
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics
    
def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    
    inds = np.where(ind == 0)
    ind = inds[1]
    
    # if ind.shape[0] > x.shape[0]:
    #     largs_ones = np.ones(x.shape[0], ind.shape[0]//x.shape[0] + 1)*1e5
    #     largs_ones[inds[0],inds[1]]= inds[1]
        
    #     ind = largs_ones.min(axis=1)
    
    
    
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def sim_rerank(original_sim, adjust_sim, K):
    '''
    rerank original_sim top K with adjust_sim
    '''
   
    K = min(original_sim.shape[1]-1, K)
    b_sim = original_sim.shape[0]
    rerank_sim = original_sim.copy()
    adjust_sim_copy = adjust_sim.copy()
    index_2 = np.argsort(-1*original_sim)[:, :K+1].reshape(-1)
    index_1 = np.arange(b_sim)[:,None].repeat(K+1,-1).reshape(-1)
    
    adjust_picked = adjust_sim[index_1, index_2].reshape(b_sim, -1)
    adjust_picked_rerank = np.argsort(-1*adjust_picked).reshape(-1) 
    index_3 = index_2.reshape(b_sim,-1)[index_1, adjust_picked_rerank]
    rerank_sim[index_1,index_3]=rerank_sim[index_1,index_2]
    
    

    return rerank_sim

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

if __name__ == '__main__':
    original_sim = np.random.rand(1,5)
    adjust_sim =np.random.rand(1,5)#

    rerank_sim = sim_rerank(original_sim, adjust_sim,3)
    

    print(rerank_sim)
    