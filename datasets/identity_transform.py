import torch


class IdentityTransform:
    def __call__(self, sample):
        sample["image"] = torch.clamp(sample["image"], 0, 1)
        return sample