import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedSampler(Sampler):
    def __init__(self, seg_dataset_len, seq_dataset_len, batch_size):
        self.seg_dataset_len = seg_dataset_len
        self.seq_dataset_len = seq_dataset_len
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "Batch size must be even."

        # Calculate how many indices to select for each type per batch
        self.seg_per_batch = self.batch_size // 2
        self.seq_per_batch = self.batch_size - self.seg_per_batch

        # Initialize indices
        self.seg_indices = np.arange(self.seg_dataset_len)
        self.seq_indices = np.arange(self.seq_dataset_len) + self.seg_dataset_len

        # Shuffle initially
        np.random.shuffle(self.seg_indices)
        np.random.shuffle(self.seq_indices)

        self.seg_pointer = 0
        self.seq_pointer = 0

    def __iter__(self):
        while True:
            batch_indices = []
            for _ in range(self.seg_per_batch):
                if self.seg_pointer >= self.seg_dataset_len:
                    # Reshuffle and reset pointer if end of dataset is reached
                    np.random.shuffle(self.seg_indices)
                    self.seg_pointer = 0
                batch_indices.append(self.seg_indices[self.seg_pointer])
                self.seg_pointer += 1

            for _ in range(self.seq_per_batch):
                if self.seq_pointer >= self.seq_dataset_len:
                    # Reshuffle and reset pointer if end of dataset is reached
                    np.random.shuffle(self.seq_indices)
                    self.seq_pointer = 0
                batch_indices.append(self.seq_indices[self.seq_pointer])
                self.seq_pointer += 1

            ret = np.empty(len(batch_indices), dtype=int)
            ret[0::2] = batch_indices[:self.seg_per_batch]
            ret[1::2] = batch_indices[self.seg_per_batch:]

            for index in ret:
                yield index
