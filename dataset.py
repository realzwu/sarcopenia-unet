import os
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk

class niiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,self.images[index])
        image = np.load(img_path)
        mask = np.load(mask_path)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class KfoldDataset(niiDataset):
    def __init__(self, image_dir, mask_dir, kfold_indices, transform=None):
        super().__init__(image_dir, mask_dir, transform)
        self.kfold_indices = kfold_indices
        self.images = [self.images[i] for i in kfold_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return super().__getitem__(index)

