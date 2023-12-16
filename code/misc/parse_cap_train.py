import json
import os
#json_file1 = 'gen_cap_train.json'
#json_file2 = 'gen_cap_train2.json'
#data1 = json.load(open(json_file1))
#data2 = json.load(open(json_file2))
data_path='/export/home/.cache/lavis/msrvtt/annotations'
#data_all = data1 + data2
json_file =os.path.join(data_path, 'gen_cap_val.json')
data_all = json.load(open(json_file))
for item in data_all:
    cap = item['caption']
    if cap.lower().startswith("based on the various descriptions provided"):
        cap = cap[cap.find(',')+1:]
        item['caption'] = cap
        print(cap)
    item['caption'] = [item['caption']]
with open(os.path.join(data_path,'gen_cap_val2.json'),'w') as f:
    json.dump(data_all, f)
