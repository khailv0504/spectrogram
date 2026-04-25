import glob
import os
import re

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from project_spectrogram.datasets.agument import RFAugment
from project_spectrogram.datasets.identity_transform import IdentityTransform
from project_spectrogram.datasets.spectrogram_dataset import SpectrogramDataset
from project_spectrogram.datasets.compose import Compose


class Preprocessing:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def parse_info(self, path):
        filename = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))  # folder name

        snr = int(re.search(r"snr(\d+)", filename).group(1))
        doppler = int(re.search(r"doppler(\d+)", filename).group(1))

        group_id = f"{label}_snr{snr}_doppler{doppler}"

        return label, snr, doppler, group_id

    def process(self):
        paths = glob.glob(os.path.join(self.root_dir, "**/*.png"), recursive=True)

        labels = sorted(list(set(os.path.basename(os.path.dirname(p)) for p in paths)))
        label2idx = {l: i for i, l in enumerate(labels)}

        print(labels)

        groups = []
        y = []

        for p in paths:
            label, snr, doppler, group_id = self.parse_info(p)
            groups.append(group_id)
            y.append(label2idx[label])

        gss = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(paths, y, groups))

        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]

        # PARSE TOÀN BỘ METADATA TRƯỚC (CHẠY 1 LẦN)
        def get_all_meta_and_labels(path_list):
            metas, labels = [], []
            for p in path_list:
                label_str, snr, doppler, _ = self.parse_info(p)
                metas.append([snr, doppler])
                labels.append(label2idx[label_str])
            return np.array(metas, dtype=np.float32), np.array(labels, dtype=np.int64)

        train_meta_raw, train_labels = get_all_meta_and_labels(train_paths)
        val_meta_raw, val_labels = get_all_meta_and_labels(val_paths)

        # CHUẨN HÓA OFFLINE
        scaler = StandardScaler()
        train_meta_scaled = scaler.fit_transform(train_meta_raw)  # Fit & Transform tập Train
        val_meta_scaled = scaler.transform(val_meta_raw)  # Chỉ Transform tập Val

        train_tf = Compose([
            RFAugment(),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        val_tf = Compose([
            IdentityTransform(),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # BƠM DATA TĨNH VÀO DATASET
        train_dataset = SpectrogramDataset(
            train_paths, train_labels,
            train_meta_scaled, transform=train_tf
        )
        val_dataset = SpectrogramDataset(
            val_paths, val_labels,
            val_meta_scaled, transform=val_tf
        )

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False
        )

        return train_loader, val_loader, labels

