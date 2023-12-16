import math
import torch


class HDFSDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=False,
        seed=0,
        drop_last_batch=False,
        drop_last_sample=True,
    ):
        super(HDFSDistributedSampler, self).__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last_sample,
        )
        # batch size on each gpu
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.drop_last_sample = drop_last_sample

    @staticmethod
    def chunk(iterable, chunk_size, drop_last_batch):
        ret = []
        for record in iterable:
            ret.append(record)
            if len(ret) == chunk_size:
                yield ret
                ret = []
        if not drop_last_batch and ret:
            yield ret

    def __iter__(self):
        iterable = super(HDFSDistributedSampler, self).__iter__()
        return HDFSDistributedSampler.chunk(
            iterable, self.batch_size, self.drop_last_batch
        )

    def __len__(self):
        if self.drop_last_batch:
            return math.floor(1.0 * self.num_samples / self.batch_size)
        else:
            return math.ceil(1.0 * self.num_samples / self.batch_size)

