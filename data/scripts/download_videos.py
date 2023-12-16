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

import pandas as pd
import os
import numpy as np
import argparse
import concurrent.futures
import warnings
import glob

# youtube-dl version 2021.12.17

def ytb_save(id, save_fp):
    os.system(f"youtube-dl -f 'bestvideo[ext=mp4][height<=360]+bestaudio/best[ext=mp4][height<=360]' -o '{save_fp}/{id}.mp4' https://www.youtube.com/watch?v={id}")


def main(args):
    ### preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    if not os.path.exists(os.path.join(video_dir, 'videos')):
        os.makedirs(os.path.join(video_dir, 'videos'))

    df = pd.read_csv(os.path.join(args.data_dir, 'annotations/20k_meta.csv'))
    print(len(df))

    df = df.drop_duplicates(subset='youtube_id', keep='first')

    ids_todo = []
    save_fps = []
    for idx, row in df.iterrows():
        video_fp = os.path.join(video_dir, str(row['youtube_id']) + '.mp4')
        if not os.path.isfile(video_fp):
            ids_todo.append(row['youtube_id'])
            save_fps.append(video_dir)
    
    print(f'Spawning {len(ids_todo)} jobs')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
        future_to_url = {executor.submit(ytb_save, url, fp) for url, fp in zip(ids_todo, save_fps)}
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Downloader')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()
    
    main(args)
