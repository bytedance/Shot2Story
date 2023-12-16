"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
import numpy as np
import random
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset


class VideoCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

class VideoAVCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, fix_total=False, audio_target=False, flexible_sampling=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        if audio_target:
            self.annotation = [ann for ann in self.annotation if ann["audio_caption"] != '']

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)
        caption = self.text_processor(ann["caption"])
        audio_caption = self.text_processor(ann["audio_caption"])
        asr = self.text_processor(ann["asr"])
        whole_asr = self.text_processor(ann["whole_ASR"])
        whole_caption = self.text_processor(ann["whole_caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "text_audio_input": audio_caption,
            "asr": asr,
            "whole_asr": whole_asr,
            "whole_caption": whole_caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

class VideoAVWholeCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, fix_total=False, audio_target=False, flexible_sampling=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.multishot = False
        self.n_frms = self.vis_processor.n_frms
        self.fix_total = fix_total

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        caption = self.text_processor(ann["whole_caption"])
        captions_shots = [self.text_processor(cap) for cap in ann["captions"]]
        
        if not self.fix_total:
            shot_names = ann["video_names"]
            num_shots = len(shot_names)
            total_frames_to_sample = self.n_frms * num_shots
            self.vis_processor.n_frms = total_frames_to_sample
            
        video_path = os.path.join(self.vis_root, vname)
        video = self.vis_processor(video_path)
        
        if not self.fix_total:
            self.vis_processor.n_frms = self.n_frms

        # audio_caption = self.text_processor(ann["audio_caption"])
        # asr = self.text_processor(ann["asr"])
        whole_asr = self.text_processor(ann["whole_ASR"])
        whole_caption = self.text_processor(ann["whole_caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "text_input_shots": captions_shots,
            "text_audio_input": '',
            "asr": '',
            "whole_asr": whole_asr,
            "whole_caption": whole_caption,
            "image_id": self.img_ids[ann["image_id"]],
            "fix_total": self.fix_total,
            "video_name": vname,
            "shot_split": [self.n_frms]*len(ann['video_names']),
        }

class VideoAVMultiCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, fix_total=False, audio_target=False, flexible_sampling=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.multishot = True
        self.n_frms = self.vis_processor.n_frms
        self.fix_total = fix_total

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        caption = self.text_processor(ann["whole_caption"])
        captions_shots = [self.text_processor(cap) for cap in ann["captions"]]
        
        shot_names = ann["video_names"]
        num_shots = len(shot_names)
        
        if self.fix_total:
            total_frames_to_sample = self.n_frms

            # Calculate the number of frames each shot should get
            frames_per_shot = total_frames_to_sample // num_shots
            # Calculate the number of remaining frames
            remaining_frames = total_frames_to_sample % num_shots
            # Create a list of frames per shot
            frames_to_sample = [frames_per_shot] * num_shots
            # Distribute the remaining frames evenly among the shots
            for i in range(remaining_frames):
                frames_to_sample[i] += 1
                
            shot_videos = []
            for n_frms, shot_name in zip(frames_to_sample, shot_names):
                self.vis_processor.n_frms = n_frms
                video_path = os.path.join(self.vis_root, shot_name)
                shot_videos.append(self.vis_processor(video_path))
            assert shot_videos[0].shape[0] == 3
            # print([s.shape for s in shot_videos])
            video = torch.cat(shot_videos, dim=1)
            self.vis_processor.n_frms = self.n_frms
        else:
            shot_videos = []
            for shot_name in shot_names:
                video_path = os.path.join(self.vis_root, shot_name)
                shot_videos.append(self.vis_processor(video_path))
            assert shot_videos[0].shape[0] == 3
            video = torch.cat(shot_videos, dim=1)
            frames_to_sample = [self.n_frms] * num_shots
        
        whole_asr = self.text_processor(ann["whole_ASR"])
        whole_caption = self.text_processor(ann["whole_caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "text_input_shots": captions_shots,
            "text_audio_input": '',
            "asr": '',
            "whole_asr": whole_asr,
            "whole_caption": whole_caption,
            "image_id": self.img_ids[ann["image_id"]],
            "shot_split": frames_to_sample,
            "video_name": vname,
            "fix_total": self.fix_total,
        }

class VideoCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": index,
            "instance_id": ann["instance_id"],
        }


class VideoAVCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, fix_total=False, audio_target=False, flexible_sampling=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print('Entering VideoAVCaptionEvalDataset')
        print(f'Audio target {audio_target}, {len(self.annotation)}')
        if audio_target:
            self.annotation = [ann for ann in self.annotation if ann["audio_caption"] != '']
        print(f'After Audio target {audio_target}, {len(self.annotation)}')

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)
        caption = self.text_processor(ann["caption"])
        audio_caption = self.text_processor(ann["audio_caption"])
        asr = self.text_processor(ann["asr"])
        whole_asr = self.text_processor(ann["whole_ASR"])
        whole_caption = self.text_processor(ann["whole_caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "text_audio_input": audio_caption,
            "asr": asr,
            "whole_asr": whole_asr,
            "whole_caption": whole_caption,
            "video_name": vname,
        }


class VideoAVMultiCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, fix_total=False, audio_target=False, flexible_sampling=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.multishot = True
        self.n_frms = self.vis_processor.n_frms
        self.fix_total = fix_total
        self.flexible_sampling = flexible_sampling
        # self.flexible_sampling_frames = [16,8,4,2,1,1]
        # self.flexible_sampling_thrs = [0,4,8,16,32,64,128]
        self.flexible_sampling_frames = [8,4,4,2,1,1]
        self.flexible_sampling_thrs = [0,2,4,8,16,32,64]
        # self.flexible_sampling_frames = [8,4,4,4]
        # self.flexible_sampling_thrs = [0,2,4,8,10]
        self.max_shots = 32
        
        self.flexible_sampling_mapping = {}
        for i in range(1, len(self.flexible_sampling_thrs)):
            for f_n in range(self.flexible_sampling_thrs[i-1], self.flexible_sampling_thrs[i]):
                self.flexible_sampling_mapping[f_n] = self.flexible_sampling_frames[i-1]

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        caption = self.text_processor(ann["whole_caption"])
        video_path = os.path.join(self.vis_root, vname)
        
        shot_names = ann["video_names"]
        num_shots = len(shot_names)
        
        if self.fix_total:
            total_frames_to_sample = self.n_frms

            # Calculate the number of frames each shot should get
            frames_per_shot = total_frames_to_sample // num_shots
            # Calculate the number of remaining frames
            remaining_frames = total_frames_to_sample % num_shots
            # Create a list of frames per shot
            frames_to_sample = [frames_per_shot] * num_shots
            # Distribute the remaining frames evenly among the shots
            for i in range(remaining_frames):
                frames_to_sample[i] += 1
                
            shot_videos = []
            for n_frms, shot_name in zip(frames_to_sample, shot_names):
                self.vis_processor.n_frms = n_frms
                video_path = os.path.join(self.vis_root, shot_name)
                shot_videos.append(self.vis_processor(video_path))
            assert shot_videos[0].shape[0] == 3
            # print([s.shape for s in shot_videos])
            video = torch.cat(shot_videos, dim=1)
            self.vis_processor.n_frms = self.n_frms
        else:
            if self.flexible_sampling:
                print('flexible_sampling')
                num_shots = min(self.max_shots, len(shot_names))
                shot_ids = random.sample(range(len(shot_names)), num_shots)
                shot_ids.sort()
                shot_names = [ann["video_names"][id] for id in shot_ids]
                n_frms = self.flexible_sampling_mapping[num_shots]
                
                shot_videos = []
                for shot_name in shot_names:
                    self.vis_processor.n_frms = n_frms
                    video_path = os.path.join(self.vis_root, shot_name)
                    shot_videos.append(self.vis_processor(video_path))
                assert shot_videos[0].shape[0] == 3
                video = torch.cat(shot_videos, dim=1)
                self.vis_processor.n_frms = self.n_frms
                frames_to_sample = [n_frms] * num_shots
            else:
                shot_videos = []
                for shot_name in shot_names:
                    video_path = os.path.join(self.vis_root, shot_name)
                    shot_videos.append(self.vis_processor(video_path))
                assert shot_videos[0].shape[0] == 3
                video = torch.cat(shot_videos, dim=1)
                frames_to_sample = [self.n_frms] * num_shots
            
        whole_asr = self.text_processor(ann["whole_ASR"])
        whole_caption = self.text_processor(ann["whole_caption"])
        
        # video = self.vis_processor(video_path)
        # # asr = self.text_processor(ann["asr"])
        # whole_asr = self.text_processor(ann["whole_ASR"])
        # whole_caption = self.text_processor(ann["whole_caption"])
        # caption = self.text_processor(ann["whole_caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "text_audio_input": '',
            "asr": '',
            "whole_asr": whole_asr,
            "whole_caption": whole_caption,
            "shot_split": frames_to_sample,
            "video_name": vname,
            "fix_total": self.fix_total,
        }
        
    def get_sample_nfrms(self, num_shots, whole_asr):
        if self.fix_total:
            total_frames_to_sample = self.n_frms

            # Calculate the number of frames each shot should get
            frames_per_shot = total_frames_to_sample // num_shots
            # Calculate the number of remaining frames
            remaining_frames = total_frames_to_sample % num_shots
            # Create a list of frames per shot
            frames_to_sample = [frames_per_shot] * num_shots
            # Distribute the remaining frames evenly among the shots
            for i in range(remaining_frames):
                frames_to_sample[i] += 1
                
            shot_ids = list(range(num_shots))
        else:
            if self.flexible_sampling:
                print('flexible_sampling')
                num_shots = min(self.max_shots, num_shots)
                shot_ids = random.sample(range(num_shots), num_shots)
                shot_ids.sort()
                n_frms = self.flexible_sampling_mapping[num_shots]
                frames_to_sample = [n_frms] * num_shots
            else:
                shot_videos = []
                shot_ids = list(range(num_shots))
                frames_to_sample = [self.n_frms] * num_shots
                
        return {
            "shot_ids": shot_ids,
            "whole_asr": whole_asr,
            "shot_split": frames_to_sample,
            "fix_total": self.fix_total,
        }
        
    
    def get_samples_template(self, num_shots, whole_asr):
        if self.fix_total:
            total_frames_to_sample = self.n_frms

            # Calculate the number of frames each shot should get
            frames_per_shot = total_frames_to_sample // num_shots
            # Calculate the number of remaining frames
            remaining_frames = total_frames_to_sample % num_shots
            # Create a list of frames per shot
            frames_to_sample = [frames_per_shot] * num_shots
            # Distribute the remaining frames evenly among the shots
            for i in range(remaining_frames):
                frames_to_sample[i] += 1
                
            shot_ids = list(range(num_shots))
            # shot_videos = []
            # for n_frms, shot_name in zip(frames_to_sample, shot_names):
            #     self.vis_processor.n_frms = n_frms
            #     video_path = os.path.join(self.vis_root, shot_name)
            #     shot_videos.append(self.vis_processor(video_path))
            # assert shot_videos[0].shape[0] == 3
            # # print([s.shape for s in shot_videos])
            # video = torch.cat(shot_videos, dim=1)
            # self.vis_processor.n_frms = self.n_frms
        else:
            if self.flexible_sampling:
                print('flexible_sampling')
                num_shots = min(self.max_shots, num_shots)
                shot_ids = random.sample(range(num_shots), num_shots)
                shot_ids.sort()
                n_frms = self.flexible_sampling_mapping[num_shots]
                frames_to_sample = [n_frms] * num_shots
            else:
                shot_videos = []
                shot_ids = list(range(num_shots))
                frames_to_sample = [self.n_frms] * num_shots
            
        whole_asr = self.text_processor(whole_asr)
        
        # video = self.vis_processor(video_path)
        # # asr = self.text_processor(ann["asr"])
        # whole_asr = self.text_processor(ann["whole_ASR"])
        # whole_caption = self.text_processor(ann["whole_caption"])
        # caption = self.text_processor(ann["whole_caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            # "video": video,
            # "text_input": caption,
            # "text_audio_input": '',
            # "asr": '',
            "shot_ids": shot_ids,
            "whole_asr": whole_asr,
            "shot_split": frames_to_sample,
            # "video_name": vname,
            "fix_total": self.fix_total,
        }


class VideoAVWholeCaptionEvalDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, fix_total=False, audio_target=False, flexible_sampling=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.multishot = False
        self.fix_total = fix_total
        self.n_frms = self.vis_processor.n_frms

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        caption = self.text_processor(ann["whole_caption"])
        
        if not self.fix_total:
            shot_names = ann["video_names"]
            num_shots = len(shot_names)
            total_frames_to_sample = self.n_frms * num_shots
            self.vis_processor.n_frms = total_frames_to_sample
            
        video_path = os.path.join(self.vis_root, vname)
        video = self.vis_processor(video_path)
        
        if not self.fix_total:
            self.vis_processor.n_frms = self.n_frms

        # audio_caption = self.text_processor(ann["audio_caption"])
        # asr = self.text_processor(ann["asr"])
        whole_asr = self.text_processor(ann["whole_ASR"])
        whole_caption = self.text_processor(ann["whole_caption"])
        
        # print('Eval', video.shape, [self.n_frms]*len(ann['video_names']), self.fix_total)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "text_audio_input": '',
            "asr": '',
            "whole_asr": whole_asr,
            "whole_caption": whole_caption,
            "image_id": self.img_ids[ann["image_id"]],
            "video_name": vname,
            "fix_total": self.fix_total,
            "shot_split": [self.n_frms]*len(ann['video_names']),
        }