import torch
import bisect
import itertools
from torch.utils.data.dataset import ConcatDataset


class ConcatHDFSDataset(ConcatDataset):
    def __getitem__(self, index_list):
        # group index_list based on self.cumulative_sizes
        dataset_ids = [ bisect.bisect_right(self.cumulative_sizes, idx) for idx in index_list]
        grouped_ids = [[] for _ in self.datasets]
        for dataset_idx, idx in zip(dataset_ids, index_list):
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            grouped_ids[dataset_idx].append(sample_idx)
        all_samples = []
        for x in range(len(self.datasets)):
            if len(grouped_ids[x]) > 0:
                batch = self.datasets[x][grouped_ids[x]]
                all_samples.append(batch)
        merged_samples = {}
        for k,v in all_samples[0].items():
            if torch.is_tensor(v):
                merged_samples[k] = torch.cat([x[k] for x in all_samples], 0)
            else:
                merged_samples[k] = list(itertools.chain.from_iterable([x[k] for x in all_samples]))
        return merged_samples

