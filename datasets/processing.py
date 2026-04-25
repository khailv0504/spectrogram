import glob
import os
import re
from typing import Any

import numpy as np
import torchaudio
import torchvision
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from project_spectrogram.datasets.spectrogram_dataset import SpectrogramDataset


class Preprocessing:
    def __init__(self, root_dir, data_config: dict[str, Any] | None = None):
        self.root_dir = root_dir
        self.data_config = data_config or {}

    def parse_info(self, path):
        filename = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))  # folder name

        snr = int(re.search(r"snr(\d+)", filename).group(1))
        doppler = int(re.search(r"doppler(\d+)", filename).group(1))

        group_id = f"{label}_snr{snr}_doppler{doppler}"

        return label, snr, doppler, group_id

    def process(self):
        paths = glob.glob(os.path.join(self.root_dir, "**/*.png"), recursive=True)
        if not paths:
            raise ValueError(f"No .png files found under: {self.root_dir}")

        labels = sorted(list(set(os.path.basename(os.path.dirname(p)) for p in paths)))
        label2idx = {l: i for i, l in enumerate(labels)}

        print(labels)

        groups = []
        y = []

        for p in paths:
            label, snr, doppler, group_id = self.parse_info(p)
            groups.append(group_id)
            y.append(label2idx[label])

        val_split = float(self.data_config.get("val_split", 0.2))
        split_random_state = int(self.data_config.get("split_random_state", 42))
        gss = GroupShuffleSplit(test_size=val_split, random_state=split_random_state)
        train_idx, val_idx = next(gss.split(paths, y, groups))

        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]

        # PARSE TOÀN BỘ METADATA TRƯỚC (CHẠY 1 LẦN)
        def get_all_labels(path_list):
            labels = []
            for p in path_list:
                label_str, _, _, _ = self.parse_info(p)
                labels.append(label2idx[label_str])
            return np.array(labels, dtype=np.int64)

        train_labels = get_all_labels(train_paths)
        val_labels = get_all_labels(val_paths)


        normalize_mean = tuple(self.data_config.get("normalize_mean", [0.5, 0.5, 0.5]))
        normalize_std = tuple(self.data_config.get("normalize_std", [0.5, 0.5, 0.5]))
        time_mask_param = int(self.data_config.get("time_mask_param", 15))
        freq_mask_param = int(self.data_config.get("freq_mask_param", 20))

        train_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
                torchvision.transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        val_tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

        # BƠM DATA TĨNH VÀO DATASET
        train_dataset = SpectrogramDataset(
            train_paths, train_labels,
            transform=train_tf
        )
        val_dataset = SpectrogramDataset(
            val_paths, val_labels,
            transform=val_tf
        )

        batch_size_train = int(self.data_config.get("batch_size_train", 64))
        batch_size_val = int(self.data_config.get("batch_size_val", 32))
        num_workers = int(self.data_config.get("num_workers", 4))
        pin_memory = bool(self.data_config.get("pin_memory", True))
        drop_last_train = bool(self.data_config.get("drop_last_train", True))
        drop_last_val = bool(self.data_config.get("drop_last_val", False))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_val,
        )

        return train_loader, val_loader, labels

