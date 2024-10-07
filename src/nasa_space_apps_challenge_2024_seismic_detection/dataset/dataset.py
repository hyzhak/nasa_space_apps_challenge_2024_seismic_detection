from collections import defaultdict

import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random

from nasa_space_apps_challenge_2024_seismic_detection.dataset.categories import SIGNAL
from nasa_space_apps_challenge_2024_seismic_detection.dataset.seismic_record import restore_record

mini_batch_size = 8  # Number of segments per mini-batch


# Custom Dataset class to handle waveform loading and segment extraction
class SeismicDataset(Dataset):
    def __init__(self, file_list, input_processed_path, segment_size, random_segment=True, segment_scan_step=None,
                 dtype=torch.float32):
        self.file_list = file_list
        self.input_processed_path = input_processed_path
        self.segment_size = segment_size
        self.dtype = dtype
        self.random_segment = random_segment
        self.segment_scan_step = segment_scan_step
        self._total_size_per_item = defaultdict(int)

    def extract_segment(self, waveform):
        start_segment = np.random.randint(0, len(waveform) - self.segment_size + 1)
        return waveform[start_segment:start_segment + self.segment_size]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        record = restore_record(os.path.join(self.input_processed_path, filename))

        segments = None
        labels = None
        # while selected_waveform is None:
        #     markup = random.choice(record.markup)
        #     labels = int(markup.category == SIGNAL)
        #     waveform = record.waveform[markup.start_idx:markup.end_idx]
        #     if len(waveform) >= self.segment_size:
        #         selected_waveform = waveform

        if self.random_segment:
            selected_waveform = None
            while selected_waveform is None:
                markup = random.choice(record.markup)
                waveform = record.waveform[markup.start_idx:markup.end_idx]
                if len(waveform) >= self.segment_size:
                    selected_waveform = waveform
                    labels = int(markup.category == SIGNAL)

            segment = self.extract_segment(selected_waveform)

            segment = normalize_waveform(segment)
            segment = torch.tensor(segment, dtype=self.dtype)
            # Add channel dimension to make the tensor shape (1, segment_size)
            segments = segment.unsqueeze(0)
        else:
            # slide window with step self.segment_scan_step over the waveform
            # and extract segments
            segments = []
            labels = []

            for markup in record.markup:
                waveform = record.waveform[markup.start_idx:markup.end_idx]
                if len(waveform) >= self.segment_size:
                    segment = self.extract_segment(waveform)
                    segment = normalize_waveform(segment)
                    segment = torch.tensor(segment, dtype=self.dtype)
                    # Add channel dimension to make the tensor shape (1, segment_size)
                    segment = segment.unsqueeze(0)

                    segments.append(segment)

                    label = int(markup.category == SIGNAL)
                    labels.append(label)

            self._total_size_per_item[idx] = len(segments)

        return segments, labels


def normalize_waveform(waveform):
    return waveform / np.max(np.abs(waveform))


# Custom collate function to handle variable-length batches
def variable_length_batches_collate_fn(batch):
    batch_segments = []
    batch_labels = []

    for segments, labels in batch:
        if isinstance(segments, list):
            batch_segments.extend(segments)
            batch_labels.extend(labels)
        else:
            batch_segments.append(segments)
            batch_labels.append(labels)

    # Now, batch_segments contains all the segments; let's split them into mini-batches
    return batch_segments, batch_labels


# Mini-batch generator that splits large batches into smaller mini-batches using yield
class MiniBatchGenerator:
    def __init__(self, data_loader, mini_batch_size):
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.mini_batch_size = mini_batch_size
        self._total_size = None

    def __len__(self):
        return self._total_size

    def __iter__(self):
        total_size = 0
        for batch_segments, batch_labels in self.data_loader:
            num_segments = len(batch_segments)
            total_size += num_segments

            # Splitting the large batch into smaller mini-batches and yielding them
            for i in range(0, num_segments, self.mini_batch_size):
                mini_batch_segments = batch_segments[i:i + self.mini_batch_size]
                mini_batch_labels = batch_labels[i:i + self.mini_batch_size]
                yield torch.stack(mini_batch_segments), torch.tensor(mini_batch_labels)

        self._total_size = total_size
