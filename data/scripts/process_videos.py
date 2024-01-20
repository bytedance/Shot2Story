# Copyright 2023 ByteDance and/or its affiliates.
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


import os
import ast
import decord
import asynchat
import argparse

import pandas as pd
import pickle as pkl
import multiprocessing as mp
from datetime import datetime, timedelta

lock = mp.Lock()

def write_to_file(filename, content):
    with lock:
        with open(filename, 'a') as f:
            f.writelines(content + '\n')

def parse_large_timestamps(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    td = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return td


def split_video(input):
    row, shots = input
    cmd_template = "ffmpeg -n -i {} -ss {} -t {} -c:v libx264 -c:a aac {}" # -c:v libx264 -c:a aac 
    
    video_name = row['video_name']
    clip_id = row['clip_id']
    
    start =ast.literal_eval(row['duration'])[0]
    end = ast.literal_eval(row['duration'])[1]
    duration = (parse_large_timestamps(end) - parse_large_timestamps(start)).total_seconds()
    cmd = cmd_template.format(f"./data/videos/{video_name}", start, duration, f"./data/videos_extracted/{clip_id}.mp4")
    os.system(cmd)
    # write_to_file(vids_file, cname)
    
    vreader = decord.VideoReader(f"./data/videos_extracted/{clip_id}.mp4")
    fps = vreader.get_avg_fps()
    for shot in shots:
        start_time = shot[0]/fps
        end_time = shot[1]/fps
        duration = end_time - start_time
        cmd = cmd_template.format(f"./data/videos_extracted/{clip_id}.mp4", start_time, duration, f"./data/videos_extracted/{clip_id}_{shot[0]}_{shot[1]}.mp4")
        os.system(cmd)
    
    return clip_id
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=1)
    args = parser.parse_args()

    data = pd.read_csv('./data/relevant_videos_exists.txt', names=['video_name'])
    data = data.sort_values(by='video_name')
    data['youtube_id'] = data['video_name'].str.rsplit('.', n=1, expand=True)[0]

    split_info = pkl.load(open('data/annotations/20k_split_info.pkl', 'rb'))
    meta_data = pd.read_csv('./data/annotations/20k_meta.csv')
    
    unavailable_data = meta_data[~meta_data['youtube_id'].isin(data['youtube_id'])]
    print(f"Unavailable data: {len(unavailable_data)}")
    print(unavailable_data.youtube_id.values)
    
    data = pd.merge(data, meta_data, on='youtube_id')
    vids_file = './data/existing_videos_split.csv'

    os.makedirs('./data/videos_extracted', exist_ok=True)
    
    try:
        exitsting_vids = [l.strip() for l in open(vids_file, 'r').readlines()]
    except:
        exitsting_vids = []
        
    inputs = []
    for index, row in data.iterrows():
        if row['clip_id']+'.mp4' in vids_file: continue
        inputs.append([row, split_info[row['clip_id']+'.mp4']])
    
    pool = mp.Pool(args.processes)
    r = pool.map(split_video, inputs)
    pool.close()
    pool.join()
    
    open('./data/existing_videos_split.csv', 'w').writelines('\n'.join(r))