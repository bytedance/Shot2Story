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
from PIL import Image
import io
from torch.utils.data import Dataset
from lavis.datasets.hdfs_utils import hload_pkl
from transformers import AutoTokenizer
try:
    from dataloader import KVReader
except ImportError:
    raise ImportError("Please install dataloader")



class HDFSImageCaptionDataset(Dataset):
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
        ids = 0
        for im_name, text in data.items():
            tmp = dict()
            tmp['filename'] = im_name
            tmp['video_ids'] = ids
            tmp['text_list'] = [text,]
            data_list.append(tmp)
            ids += 1

        return data_list
    
    def __len__(self):
        return len(self.data_infors)
    
    def __getitem__(self, index_list):
        all_res = [self.data_infors[index] for index in index_list]
        all_ims = [item['filename'] for item in all_res]
        all_text = [item['text_list'][0] for item in all_res]
        image_bins=self.reader.read_many(all_ims)
        #rand_text = random.choice(results['text_list'])
        #results['text']=rand_text

        #return self.pipeline(results)
        
        images = [ Image.open(io.BytesIO(image_bin)).convert("RGB") for image_bin in image_bins]
        images = [self.vis_processor(image) for image in images]
        captions = [self.text_processor(text) for text in all_text]

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": torch.stack(images, 0),
            "text_input": captions, #torch.stack(captions, 0),
            "image_id": index_list,
        }

