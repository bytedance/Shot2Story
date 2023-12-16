import copy
import json
import os.path as osp
import random
from collections import defaultdict
import numpy as np
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from torchvision.transforms import Compose
import torch
from torch.utils.data import Dataset
from lavis.datasets.hdfs_utils import hload_pkl
from transformers import AutoTokenizer
try:
    from dataloader import KVReader
except ImportError:
    raise ImportError("Please install dataloader")

try:
    from nlgeval import NLGEval
except Exception:
    print("pip3 install git+https://github.com/Maluuba/nlg-eval.git@master --user ")


class HDFSVideoCaptionDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__()#vis_processor, text_processor, vis_root, ann_paths)
        self.data_root = vis_root
        self.data_infors = self.load_annotations(ann_paths[0])
        sys.stdout.flush()
        self.data_type = 'hdfs'
        self.num_readers = 2

        #self.reader = KVReader(self.data_root, self.num_readers)
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def load_annotations(self,data_ann):
        data = hload_pkl(data_ann)
        data_list=[]

        for ids, item in enumerate(data):
            filename=item['filename']
            text = item['text']
            tmp = dict()
            tmp['filename'] = filename
            tmp['video_ids'] = ids
            tmp['text_list'] = [text,]
            data_list.append(tmp)

        return data_list
    
    def __len__(self):
        return len(self.data_infors)
    
    def __getitem__(self, index_list):
        all_res = [self.data_infors[index] for index in index_list]
        all_vids = [item['filename'] for item in all_res]
        all_text = [item['text_list'][0] for item in all_res]
        video_bins=self.reader.read_many(all_vids)
        #rand_text = random.choice(results['text_list'])
        #results['text']=rand_text

        #return self.pipeline(results)
        
        videos = [self.vis_processor(video_bin) for video_bin in video_bins]
        captions = [self.text_processor(text) for text in all_text]

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": torch.stack(videos, 0),
            "text_input": captions, #torch.stack(captions, 0),
            "image_id": index_list,
        }

class HDFSVideoCaptionEvalDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__()#vis_processor, text_processor, vis_root, ann_paths)
        self.data_root = vis_root
        self.data_infors = self.load_annotations(ann_paths)

        self.data_type = 'video'
        self.num_readers = 8

        self.reader = KVReader(self.data_root, self.num_readers)
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.labels = defaultdict(list)
        for item in self.data_infors:
            self.labels[item['video_ids']].extend(item['text_list'])


    def load_annotations(self,data_ann):
        data = hload_pkl(data_ann)
        data_list=[]

        ids=0
        for item in data:
            tmp = dict()
            tmp['filename'] = item['filename']
            tmp['video_ids']=ids
            tmp['text_list']=copy.deepcopy(item['text'])
            data_list.append(tmp)
            ids = ids+1
        return data_list
    
    def __len__(self):
        return len(self.data_infors)
    def __getitem__(self, index):
        results = self.data_infors[index]
        video_bin=self.reader.read_many([results['filename'],])[0]
        rand_text = random.choice(results['text_list'])
        #results['text']=rand_text

        #return self.pipeline(results)
        
        video = self.vis_processor(video_bin)
        caption = self.text_processor(rand_text)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": index,
        }


