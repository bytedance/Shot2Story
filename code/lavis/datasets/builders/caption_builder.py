"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoAVCaptionDataset,
    VideoAVMultiCaptionDataset,
    VideoAVWholeCaptionDataset,
    VideoCaptionEvalDataset,
    VideoAVCaptionEvalDataset,
    VideoAVMultiCaptionEvalDataset,
    VideoAVWholeCaptionEvalDataset,
)

from lavis.datasets.datasets.hdfs_video_caption_datasets import (
    HDFSVideoCaptionDataset,
    HDFSVideoCaptionEvalDataset,
)
from lavis.datasets.datasets.hdfs_image_caption_datasets import (
    HDFSImageCaptionDataset,
)
@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }

@registry.register_builder("webvid_caption")
class WebVidCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = HDFSVideoCaptionDataset
    eval_dataset_cls = HDFSVideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/defaults_cap.yaml",
    }

@registry.register_builder("conceptual_caption_3m_hdfs")
class WebVidCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = HDFSImageCaptionDataset
    eval_dataset_cls = HDFSVideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m_hdfs.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }

@registry.register_builder("msrvtt_gen_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/gen_cap.yaml",
    }

@registry.register_builder("msrvtt_minigpt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/minigpt4_cap.yaml",
    }

@registry.register_builder("msrvtt_minigpt_caption_fake")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoAVCaptionDataset
    eval_dataset_cls = VideoAVCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/minigpt4_cap_fake.yaml",
    }
    
    def build(self):
        return self.build_custom_splits()

@registry.register_builder("bdmsvdc_minigpt_caption")
class BDMSVDCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoAVCaptionDataset
    eval_dataset_cls = VideoAVCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bdmsvdc/minigpt4_cap.yaml",
    }
    
    def build(self):
        return self.build_custom_splits()


@registry.register_builder("bdmsvdc_multishot_minigpt_caption")
class BDMSVDCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoAVMultiCaptionDataset
    eval_dataset_cls = VideoAVMultiCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bdmsvdc_multishot/minigpt4_cap.yaml",
    }
    
    def build(self):
        return self.build_custom_splits()

@registry.register_builder("msrvtt_multishot_minigpt_caption")
class BDMSVDCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoAVMultiCaptionDataset
    eval_dataset_cls = VideoAVMultiCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt_multishot/minigpt4_cap.yaml",
    }
    
    def build(self):
        return self.build_custom_splits()

@registry.register_builder("bdmsvdc_whole_minigpt_caption")
class BDMSVDCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoAVWholeCaptionDataset
    eval_dataset_cls = VideoAVWholeCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bdmsvdc_whole/minigpt4_cap.yaml",
    }
    
    def build(self):
        return self.build_custom_splits()

@registry.register_builder("lsmdc_minigpt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/lsmdc/minigpt4_cap.yaml",
    }

@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }
