import os
import sys
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from src.io.psee_loader import PSEELoader


class PseeDataset(Dataset):

    def __init__(self, root):
        self.paths = glob(os.path.join(root, "**/*.dat"), recursive=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        reader = PSEELoader(path)
        events = reader.load_n_events(reader.event_count())
        return events


def collate_fn(batch):

    events, lengths = [], []
    features = ['x', 'y', 't', 'p']
    for ev in batch:
        events.append(np.stack([ev[f].astype(np.float32)
                                for f in features], axis=-1))
        lengths.append(ev['x'].shape[0])

    max_length = max(lengths)
    events = [np.pad(ev, ((0, max_length-ln), (0, 0)), mode='constant')
              for ln, ev in zip(lengths, events)]
    events = torch.as_tensor(np.stack(events, axis=0))
    lengths = torch.as_tensor(lengths)

    return lengths, events
