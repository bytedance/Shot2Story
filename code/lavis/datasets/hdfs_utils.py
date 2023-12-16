import time
from typing import IO, Any, List

import shutil
import subprocess
from contextlib import contextmanager
import os
import glob
import threading
import torch
import io
import collections
from PIL import Image
import cv2
import numpy as np
import pickle
import json


HADOOP_BIN = 'HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs'

__all__ = [
    'hlist_files', 'hopen', 'hexists', 'hmkdir', 'hglob', 'hisdir', 'hcountline', 'hcopy', 'hmget',
    'hload_pkl', 'hload_vocab', 'hload_pil', 'hload_cv2', 'hload_torch',
    'load_from_hdfs', 'hsave_pkl', 'hdelete', 'hmove', 'hload_json'
]


def hload_pkl(filepath):
    """ load pickle from hdfs """
    if not filepath.startswith('hdfs://'):
        with open(filepath, 'rb') as fr:
            file = pickle.load(fr)
            return file
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        file = pickle.load(accessor)

        del accessor
        return file


def hsave_pkl(pkl_file, filepath):
    """ load pickle from hdfs """
    if filepath.startswith('hdfs://'):
        with hopen(filepath, "wb") as writer:
            pickle.dump(pkl_file, writer)
    else:
        pickle.dump(pkl_file, filepath)


def hload_json(filepath):
    if not filepath.startswith('hdfs://'):
        with open(filepath, 'r') as fr:
            file = json.load(fr)
            return file
    with hopen(filepath, 'r') as reader:
        accessor = io.BytesIO(reader.read())
        file = json.load(accessor)

        del accessor
        return file




def hload_pil(filepath):
    """
    load image from hdfs using PIL
    """
    if not filepath.startswith('hdfs://'):
        img = Image.open(filepath)
        return img
    with hopen(filepath) as f:
        image_str = f.read()
        img = Image.open(io.BytesIO(image_str)).convert("RGB")

        del image_str
        return img


def hload_cv2(filepath):
    """
    load image from hdfs using cv2
    """
    if not filepath.startswith('hdfs://'):
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img
    with hopen(filepath) as f:
        image_str = f.read()
        img = cv2.imdecode(np.frombuffer(image_str, np.uint8), cv2.IMREAD_COLOR)
        # covert BRG to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        del image_str
        return img


def hload_vocab(vocab_path):
    """Loads a vocabulary file into a dictionary from hdfs."""
    vocab = collections.OrderedDict()
    index = 0
    if vocab_path.startswith('hdfs://'):
        with hopen(vocab_path, "r") as reader:
            accessor = io.BytesIO(reader.read())
            while True:
                token = accessor.readline()
                token = token.decode('utf-8')  # 要解码使得数据接口类型一致
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
            del accessor
            return vocab
    else:
        with open(vocab_path, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
            return vocab


def hload_torch(filepath: str, map_location='cpu', **kwargs):
    """ load model from hdfs """
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, map_location=map_location, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, map_location=map_location, **kwargs)
        del accessor
        return state_dict






@contextmanager  # type: ignore
def hopen(hdfs_path: str, mode: str = "r", **kwargs) -> IO[Any]:
    """
        打开一个 hdfs 文件, 用 contextmanager.

        Args:
            hfdfs_path (str): hdfs文件路径
            mode (str): 打开模式，支持 ["r", "w", "wa"]
    """
    pipe = None
    if mode.startswith("r"):
        try:
            pipe = subprocess.Popen(
                "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE, **kwargs)
            yield pipe.stdout
            pipe.stdout.close()  # type: ignore
            pipe.wait()
            return
        except Exception:
            time.sleep(10)
            pipe = subprocess.Popen(
                "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE, **kwargs)
            yield pipe.stdout
            pipe.stdout.close()  # type: ignore
            pipe.wait()
            return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE, **kwargs)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE, **kwargs)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hlist_files(folders: List[str]) -> List[str]:
    """
        罗列一些 hdfs 路径下的文件。

        Args:
            folders (List): hdfs文件路径的list
        Returns:
            一个list of hdfs 路径
    """
    if isinstance(folders, str):
        folders = [folders]
    files = []
    for folder in folders:
        if folder.startswith('hdfs'):
            pipe = subprocess.Popen("{} dfs -ls {}".format(HADOOP_BIN, folder), shell=True,
                                    stdout=subprocess.PIPE)
            # output, _ = pipe.communicate()
            for line in pipe.stdout:  # type: ignore
                line = line.strip()
                # drwxr-xr-x   - user group  4 file
                if len(line.split()) < 5:
                    continue
                files.append(line.split()[-1].decode("utf8"))
            pipe.stdout.close()  # type: ignore
            pipe.wait()
        else:
            if os.path.isdir(folder):
                files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
            elif os.path.isfile(folder):
                files.append(folder)
            else:
                print('Path {} is invalid'.format(folder))

    return files


def get_latest_checkpoint(folders: str) -> str:
    path = None
    try:
        if hexists(folders):
            ckpt_list = hlist_files(folders)
            ckpt_list = [ckpt for ckpt in ckpt_list if '_old' not in os.path.basename(ckpt)]
            epochs = [int(os.path.splitext(os.path.basename(c))[0].split('epoch_')[-1]) for c in ckpt_list]
            newest = sorted(epochs)[-1]
            path = ckpt_list[epochs.index(newest)]
    except Exception as e:
        print(e)
    return path


def hexists(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        return os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path)) == 0
    return os.path.exists(file_path)


def hdelete(file_path: str):
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        os.system("{} dfs -rm {}".format(HADOOP_BIN, file_path))
    else:
        os.remove(file_path)


def hisdir(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is a dir """
    if file_path.startswith('hdfs'):
        flag1 = os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path))  # 0:路径存在
        flag2 = os.system("{} dfs -test -f {}".format(HADOOP_BIN, file_path))  # 0:是文件 1:不是文件
        flag = ((flag1 == 0) and (flag2 != 0))
        return flag
    return os.path.isdir(file_path)


def hmkdir(file_path: str) -> bool:
    """ hdfs mkdir """
    if file_path.startswith('hdfs'):
        os.system("{} dfs -mkdir -p {}".format(HADOOP_BIN, file_path))
    else:
        os.mkdir(file_path)
    return True


def hcopy(from_path: str, to_path: str) -> bool:
    """ hdfs copy """
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            os.system("{} dfs -cp -f {} {}".format(HADOOP_BIN, from_path, to_path))
        else:
            os.system("{} dfs -copyFromLocal -f {} {}".format(HADOOP_BIN, from_path, to_path))
    else:
        if from_path.startswith("hdfs"):
            os.system("{} dfs -text {} > {}".format(HADOOP_BIN, from_path, to_path))
        else:
            shutil.copy(from_path, to_path)
    return True


def hglob(search_path, sort_by_time=False):
    """ hdfs glob """
    if search_path.startswith("hdfs"):
        if sort_by_time:
            hdfs_command = HADOOP_BIN + ' dfs -ls %s | sort -k6,7' % search_path
        else:
            hdfs_command = HADOOP_BIN + ' dfs -ls %s' % search_path
        path_list = []
        files = os.popen(hdfs_command).read()
        files = files.split("\n")
        for file in files:
            if 'hdfs' in file:
                startindex = file.index('hdfs')
                path_list.append(file[startindex:])
        return path_list
    else:
        files = glob.glob(search_path)
        if sort_by_time:
            files = sorted(files, key=lambda x: os.path.getmtime(x))
    return files


def htext_list(files, target_folder):
    for fn in files:
        name = fn.split('/')[-1]
        hdfs_command = HADOOP_BIN + ' dfs -text %s > %s/%s' % (fn, target_folder, name)
        os.system(hdfs_command)


def hmove(src_path, res_path):
    """
    将src_path移动至res_path
    """
    if src_path.startswith('hdfs://'):
        os.system(f'{HADOOP_BIN} dfs -mv {src_path} {res_path}')
    else:
        os.rename(src_path, res_path)


def hmget(files, target_folder, num_thread=16):
    """ 将整个hdfs 文件夹 get下来，但是不是简单的get，因为一些hdfs文件是压缩的，需要解压"""
    part = len(files) // num_thread
    thread_list = []
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:
            end = len(files)
        else:
            end = start + part
        t = threading.Thread(target=htext_list, kwargs={
                             'files': files[start:end], 'target_folder': target_folder})
        thread_list.append(t)

    for t in thread_list:
        t.setDaemon(True)
        t.start()

    for t in thread_list:
        t.join()


def hcountline(path):
    '''
    count line in file
    '''
    count = 0
    if path.startswith('hdfs'):
        with hopen(path, 'r') as f:
            for line in f:
                count += 1
    else:
        with open(path, 'r') as f:
            for line in f:
                count += 1
    return count

