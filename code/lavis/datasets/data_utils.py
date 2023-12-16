"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gzip
import logging
import os
import random
import random as rnd
import tarfile
import zipfile
import io
import decord
import webdataset as wds
import numpy as np
import torch
from decord.ndarray import NDArray
from torch.utils.data.dataset import IterableDataset, ChainDataset
from decord import VideoReader
from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import ConcatDataset
from lavis.datasets.datasets.concat_hdfs_datasets import ConcatHDFSDataset
from tqdm import tqdm

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")


def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", clips=None):
    # check video_path
    if type(video_path) is not str:
        file_obj = io.BytesIO(video_path)
        vr = VideoReader(file_obj, height=height, width=width)
    else:
        vr = VideoReader(uri=video_path, height=height, width=width)
    total_len = len(vr)

    if clips is not None:
        frms = []
        fps = vr.get_avg_fps()
        for clip in clips:
            strt_senconds, end_seconds, shot_strt_frms, shot_end_frms = clip
            if shot_strt_frms == 0 and shot_end_frms == -1:
                start = strt_senconds * fps + shot_strt_frms
                end = end_seconds * fps + shot_strt_frms
                vlen = end - start
            else:
                vlen = shot_end_frms - shot_strt_frms
                start = strt_senconds * fps + shot_strt_frms
                end = strt_senconds * fps + shot_end_frms
            if end > total_len:
                print(f'Video starts from {start} to {end} exceeding total len {total_len}')
                end = total_len
                vlen = end - start
            if n_frms > vlen:
                indices = np.arange(start, end, vlen / n_frms).astype(int)
            elif sampling == "uniform":
                indices = np.arange(start, end, vlen / n_frms).astype(int)
            elif sampling == "headtail":
                half = n_frms // 2
                another_half = n_frms - half
                sampled_cnt = [half, another_half]
                random.shuffle(sampled_cnt)
                indices_h = sorted(rnd.sample(range(vlen // 2), sampled_cnt[0]))
                indices_t = sorted(rnd.sample(range(vlen // 2, vlen), sampled_cnt[1]))
                indices = indices_h + indices_t
            else:
                raise NotImplementedError
            frms.append(vr.get_batch(indices).permute(3,0,1,2).float())
    else:
        vlen = len(vr)
        #print('video len', vlen)
        start, end = 0, vlen
        #n_frms = min(n_frms, vlen)
        if n_frms > vlen:
            indices = np.arange(start, end, vlen / n_frms).astype(int)
        elif sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frms).astype(int)
        elif sampling == "headtail":
            half = n_frms // 2
            another_half = n_frms - half
            sampled_cnt = [half, another_half]
            random.shuffle(sampled_cnt)
            indices_h = sorted(rnd.sample(range(vlen // 2), sampled_cnt[0]))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), sampled_cnt[1]))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        # get_batch -> T, H, W, C
        #print(frms)
        #print(frms.shape)
        indices = [int(i) if int(i) < len(vr) else vlen-1 for i in indices]
        indices = sorted(indices)[:n_frms]
        try:
            frms = vr.get_batch(indices)
            if isinstance(frms, torch.Tensor):
                frms = frms.permute(3,0,1,2).float() 
            elif isinstance(frms, NDArray):
                frms = torch.from_numpy(frms.asnumpy()).permute(3,0,1,2).float() 
        except Exception as e:
            print(indices, len(vr), n_frms)
            print(video_path)
            indices = [int(i) if int(i) < len(vr) else rnd.sample(range(vlen),1)[0] for i in indices]
            print(indices)
            print(e)
        assert len(frms[0])==n_frms, f"{frms.shape}, {len(frms)}, {indices}, {vlen}, {n_frms}"
        # frms = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        elif len(datasets[split_name]) == 1 and getattr(datasets[split_name][0], 'data_type',
                None) == 'hdfs':
            logging.info(
                    "Dataset {} is hdfs dataset, cannot be concatenated".format(
                        datasets[split_name][0])
            )
            datasets[split_name] = datasets[split_name][0]

        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            chained_datasets = (
                ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
            )
            if len(map_datasets) > 0:
                if getattr(map_datasets[0], 'data_type', None) == 'hdfs':
                    concat_datasets = ConcatHDFSDataset(map_datasets)
                    concat_datasets.data_type = map_datasets[0].data_type
                else:

                    concat_datasets = ConcatDataset(map_datasets) 
            else:
                concat_datasets = None
            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.

    Args:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {} to {}.".format(from_path, to_path))
        with tarfile.open(from_path, "r") as tar:
            files = []
            for file_ in tqdm(tar):
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            logging.info("Finished extracting tar file {}.".format(from_path))
            return files

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {} to {}.".format(from_path, to_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = []
            for file_ in tqdm(zfile.namelist()):
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        logging.info("Finished extracting zip file {}.".format(from_path))
        return files

    elif from_path.endswith(".gz"):
        logging.info("Opening gz file {} to {}.".format(from_path, to_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, "rb") as gzfile, open(filename, "wb") as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        logging.info("Finished extracting gz file {}.".format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives."
        )


def save_frames_grid(img_array, out_path):
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    if len(img_array.shape) == 3:
        img_array = img_array.unsqueeze(0)
    elif len(img_array.shape) == 5:
        b, t, c, h, w = img_array.shape
        img_array = img_array.view(-1, c, h, w)
    elif len(img_array.shape) == 4:
        pass
    else:
        raise NotImplementedError(
            "Supports only (b,t,c,h,w)-shaped inputs. First two dimensions can be ignored."
        )

    assert img_array.shape[1] == 3, "Exepcting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)
