import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, metadata=None, transform=None):
        self.paths = paths
        self.labels = torch.from_numpy(labels).long()
        self.metadata = torch.from_numpy(metadata).float()
        self.transform = transform


    def __getitem__(self, idx):

        img = read_image(self.paths[idx]).float() / 255.0

        sample = {
            "image": img,
            "meta": self.metadata[idx],
            "label": self.labels[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.paths)