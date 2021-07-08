import torch
from torch.utils.data import RandomSampler, DataLoader


class CustomDataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = range(len(dataset))

    def __len__(self):
        batch = self._reset_batch()
        for idx in self.sampler:
            batch = {
                key: torch.cat([batch[key], self.dataset[idx][key]])
                for key in self.dataset[idx].keys()
            }
            while batch["obs"].shape[0] >= self.batch_size:
                if batch["obs"].shape[0] == self.batch_size:
                    yield batch
                    batch = self._reset_batch()
                else:
                    return_batch = {
                        key: batch[key][: self.batch_size]
                        for key in self.dataset[idx].keys()
                    }
                    batch = {
                        key: batch[key][self.batch_size :]
                        for key in self.dataset[idx].keys()
                    }
                    yield return_batch
        # last batch
        if batch["obs"].shape[0] > 0 and not self.drop_last:
            yield batch

    def _reset_batch(self):
        return {key: torch.Tensor() for key in self.dataset[0].keys()}

    def __iter__(self):
        batch = self._reset_batch()
        for idx in self.sampler:
            batch = {
                key: torch.cat([batch[key], self.dataset[idx][key]])
                for key in self.dataset[idx].keys()
            }
            while batch["obs"].shape[0] >= self.batch_size:
                if batch["obs"].shape[0] == self.batch_size:
                    yield batch
                    batch = self._reset_batch()
                else:
                    return_batch = {
                        key: batch[key][: self.batch_size]
                        for key in self.dataset[idx].keys()
                    }
                    batch = {
                        key: batch[key][self.batch_size :]
                        for key in self.dataset[idx].keys()
                    }
                    yield return_batch
        # last batch
        if batch["obs"].shape[0] > 0 and not self.drop_last:
            yield batch
