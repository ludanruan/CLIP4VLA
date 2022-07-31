# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qualitative evaluation of the results.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""

import itertools
import logging
import os
import pdb
import pathlib
import shutil
from pathlib import Path

from . import html_utils as html
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
  """Visualizer class."""

  def __init__(self, name, web_dirs,  num_samples=20):
    self.name = name
    self.web_dirs = web_dirs
    
    self.num_samples = num_samples
    logger.debug("create web directories %s...", str(self.web_dirs))
    self.mkdirs(self.web_dirs)
    self.relative_dir = '../../../../'

  def mkdirs(self, paths):          
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
          if not os.path.exists(path):
            self.makedirs(path)
    else:
      self.makedirs(paths)

  def makedirs(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
  
  def _get_sample(self, dists, size, type='order'):
    if type == 'random':
      candidates= np.arange(dists.shape[0])
    elif type == 'order':
      candidates= np.arange(dists.shape[0])
      return candidates[:size]

    elif type == 'wrong':
      sorted_dists = np.sort(dists, axis=1)
      eye = np.diagonal(dists) 
      candidates = np.where(eye - sorted_dists[:,0]!=0)
      
    elif type == 'true':
      sorted_dists = np.sort(dists, axis=1)
      eye = np.diagonal(dists) 
      candidates = np.where(eye - sorted_dists[:,0]==0)

    elif  type == 'pick_from_dists':
      candidates = dists
      
    return np.random.choice(candidates,
                        size=size,
                        replace=False)
  
  def visualize_caption(self, args, results, meta, metrics, modalities, subdir_name):
    """
    Create html page to visualize caption results.
    results: list contains all generated captions. 
    meta:{"video_path":[], "captions":[]}
    metrics:{"Bleu_1":,"Bleu_2":,"Bleu_3":,"Bleu_4":,"METEOR":,"ROUGE_L":,"CIDEr":}
    """
    size = min(len(results), 5 * self.num_samples)
    sample_index = np.random.choice(len(results),
                        size=size,
                        replace=False)
    
    web_dir = os.path.join(self.web_dirs, subdir_name)
    if pathlib.Path(web_dir).exists():
      try:
        shutil.rmtree(web_dir)
      except OSError as e:
        print("Error: %s : %s" % (web_dir, e.strerror))

    if not pathlib.Path(web_dir).exists():
      pathlib.Path(web_dir).mkdir(exist_ok=True, parents=True)
    
    filepath = pathlib.Path(web_dir) / "index.html"
    if filepath.exists():
      filepath.unlink()
    pathlib.Path(web_dir).mkdir(exist_ok=True, parents=True)

    print(f"updating webpage at {web_dir}")
    title = f"Experiment name = {self.name}"
    refresh = True
    if not refresh:
      logger.debug("DISABLING WEB PAGE REFRESH")
    webpage = html.HTML(web_dir=web_dir, title=title, refresh=refresh)

    msg = f" - {self.name}"
    webpage.add_header(msg)
   
    msg = (f"BLEU1: {metrics['Bleu_1']:.3f}, "
           f"BLEU2: {metrics['Bleu_2']:.3f}, "
           f"BLEU3: {metrics['Bleu_3']:.3f}, "
           f"BLEU4: {metrics['Bleu_4']:.3f}, "
           f"METEOR: {metrics['METEOR']:.3f}, "
           f"ROUGE_L: {metrics['ROUGE_L']:.3f}, "
           f"CIDEr: {metrics['CIDEr']:.3f}")
    webpage.add_header(msg)
    logger.debug(" %d captions visualized ", size)
    vids, txts, links = [], [], []
    for ind, idx in enumerate(sample_index):
      
      
      gt_vid_path = Path(os.path.join(self.relative_dir, meta["video_path"][idx]))
      gt_caption = meta["captions"][idx]
      gt_caption.replace(" ##", "")
      result = results[idx]
      result.replace(" ##", "")

      
      txt = (f"<b>{ind + 1}<br>gt_caption:{gt_caption}<br><b>result:{result}")

      txts.append(txt)
      links.append(gt_vid_path)
      vids.append(gt_vid_path)

    
      if ind >0 and ind % 5 == 0:
       
        webpage.add_videos(vids, txts, links, width=200)
        vids, txts, links = [], [], []
    logger.debug("added %d videos", len(vids))
    webpage.save()

    
    
    return
  
  def visualize_ranking(self, args, sims, query_masks,  meta, nested_metrics,
                        modalities, subdir_name, choose_index=None):
    """Create html page to visualize retrieval results.
      sim:[batchsize,batchsize]
      query_masks: [batchsize,1]
      meta: dict_keys(['paths', 'raw_captions', 'vid_weights', 'text_weights', 'token_ids'])
            ['video_path', 'captions', 'vid_weights', 'text_weights']

            其中， paths 是batchsize个视频的路径, raw captions 是batchsize个文本 query， vid_weights[batchsize, 7]
            text_weights[batchsize,batchsize,2], token_ids:[1000,30,2]?
      modalities:['face', 'ocr', 'rgb', 's3d', 'scene', 'speech', 'vggish']
      subdir_name:self.web_dirs 下的子文件夹
      nested_metrics: metrics, including R@1, R@5...

    """
    
    query_masks = query_masks.reshape(-1).astype(bool)
    nb_queries = sims.shape[0]
    nb_candidates = sims.shape[1]
    eye = np.identity(nb_candidates, dtype=float).astype(bool)
    queries_per_candidate = nb_queries / nb_candidates
    pos_mask = np.repeat(eye, queries_per_candidate, axis=0)

    # Remove the invalid captions
    pos_mask = pos_mask[query_masks]
    sims = sims[query_masks]
    meta["captions"] = list(
        itertools.compress(meta["captions"], query_masks))
   
    dists = -sims
    gt_dists = dists[pos_mask]

    np.random.seed(0)
    sorted_ranks = np.argsort(dists, axis=1)
    rankings = []
    vis_top_k = 5
    hide_gt = False
    size = min(dists.shape[0], self.num_samples)
    
    if isinstance(choose_index, list):
      sample = np.array(choose_index)
    elif isinstance(choose_index, np.ndarray):
      sample = choose_index
    else:
      sample = np.arange(dists.shape[0])
    
    sample = self._get_sample(sample, size)
   
    for ii in sample:
      ranked_idx = sorted_ranks[ii][:vis_top_k]
      gt_captions = meta["captions"][ii]
      # ids = meta["token_ids"][ii][:, 0].numpy()
      # gt_captions = tokenizer.convert_ids_to_tokens(ids)
      gt_candidate_idx = np.where(pos_mask[ii])[0][0]
      
     
      
      datum = {
          "gt-sim": -gt_dists[ii],
          "gt-captions": gt_captions,
          "gt-rank": np.where(sorted_ranks[ii] == gt_candidate_idx)[0][0],
          "gt-path":  meta['video_path'][gt_candidate_idx],
          #0.5 * np.ones(len(modalities)), #meta["text_weights"][ii],
          "ranked_idx": ranked_idx,
          "top-k-vid_weights": meta['vid_weights'][ii][ranked_idx], #0.5 * np.ones((ranked_idx.shape[0],len(modalities))),# np.array(meta["vid_weights"])[ranked_idx],
          "top-k-text_weights": meta['text_weights'][ii][ranked_idx],
          "top-k-sims": -dists[ii][ranked_idx],
          "top-k-paths": np.array(meta["video_path"])[ranked_idx],
          "hide-gt": hide_gt,
      }
      rankings.append(datum)

    # for web_dir in self.web_dirs:
    web_dir = os.path.join(self.web_dirs, subdir_name)
    if pathlib.Path(web_dir).exists():
      try:
        shutil.rmtree(web_dir)
      except OSError as e:
        print("Error: %s : %s" % (web_dir, e.strerror))
    if not pathlib.Path(web_dir).exists():
      pathlib.Path(web_dir).mkdir(exist_ok=True, parents=True)

    self.display_retrieval_results(
        rankings,
        metrics=nested_metrics,
        modalities=modalities,
        web_dir=web_dir,
      )

  def display_retrieval_results(self, rankings, metrics, modalities,
                              web_dir):
    """Create html page to visualize the rankings."""
    visualize_weights = True

    filepath = pathlib.Path(web_dir) / "index.html"
    if filepath.exists():
      filepath.unlink()
    pathlib.Path(web_dir).mkdir(exist_ok=True, parents=True)

    print(f"updating webpage at {web_dir}")
    title = f"Experiment name = {self.name}"
    refresh = True
    if not refresh:
      logger.debug("DISABLING WEB PAGE REFRESH")
    webpage = html.HTML(web_dir=web_dir, title=title, refresh=refresh)

    msg = f" - {self.name}"
    webpage.add_header(msg)
   
    msg = (f"R1: {metrics['R1']:.3f}, "
           f"R5: {metrics['R5']:.3f}, "
           f"R10: {metrics['R10']:.3f}, "
           f"MedR: {metrics['MR']}")
    webpage.add_header(msg)
    logger.debug("Top %d retrieved videos ", len(rankings[0]))

    for line_nb, ranking in enumerate(rankings):
      vids, txts, links = [], [], []
      gt_vid_path = Path(os.path.join(self.relative_dir,ranking["gt-path"]))
      gt_captions = ranking["gt-captions"]
      gt_captions.replace(" ##", "")

      if ranking["hide-gt"]:
        txts.append(gt_captions)
        links.append("hidden")
        vids.append("hidden")
      else:
        txt = (f"<b>{line_nb + 1}<br>{gt_captions}<br><b>Rank: "
               f"{ranking['gt-rank'] + 1}, Sim: {ranking['gt-sim']:.3f} "
               f"[{gt_vid_path.stem}]")

        txts.append(txt)
        links.append(gt_vid_path)
        vids.append(gt_vid_path)

      for idx, (path, sim, text_weights, vid_weights) in enumerate(
          zip(ranking["top-k-paths"], ranking["top-k-sims"],ranking["top-k-text_weights"],
              ranking["top-k-vid_weights"])):
        path= Path(os.path.join(self.relative_dir,path))
        if ranking["hide-gt"]:
          txt = f"choice: {idx}"
        else:
          txt = f"<b>Rank: {idx + 1}, Sim: {sim:.3f}, [{path.stem}]"

        if visualize_weights:
          txt = txt + "<br><b>video weights:"
          for mod_idx, vid_weight in enumerate(vid_weights):
            mod_name = modalities[mod_idx]
            txt = txt + f"<br><b>{mod_name}: {vid_weight:.2f}"
          txt = txt + "<br><b>query weights:"
          for mod_idx, text_weight in enumerate(text_weights):
            mod_name = modalities[mod_idx]
            txt = txt + f"<br><b>{mod_name}: {text_weight:.2f}"

        txts.append(txt)
        vid_path = str(path)
        vids.append(vid_path)
        links.append(vid_path)
      webpage.add_videos(vids, txts, links, width=200)
    logger.debug("added %d videos", len(vids))
    webpage.save()
