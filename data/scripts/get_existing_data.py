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


import glob
import csv
import pandas as pd

all_files_dump = []

all_files = glob.glob('./data/videos/*.mp4')
all_files_dump.extend([f.split('/')[-1] for f in all_files])
all_files = glob.glob('./data/videos/*.mkv')
all_files_dump.extend([f.split('/')[-1] for f in all_files])

all_files_dump = [{'video_id':file.split('/')[-1][:-4], 'video_name':file.split('/')[-1]} for file in all_files_dump]
all_vids_dump = set([l['video_name'] for l in all_files_dump])
with open('./data/relevant_videos_exists.txt','w') as f:
    f.writelines('\n'.join(all_vids_dump))
# pd.DataFrame(all_files_dump).to_csv('/mnt/bn/kinetics-lp-maliva-v6/data/hdvila/data/relevant_videos_exists_w_pagedirs.csv',index=False)