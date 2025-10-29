import cv2
import numpy as np
import torch
from .config import CFG


def load_img(image_path, mask_path, scale = True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (CFG.img_size[1], CFG.img_size[0]), interpolation=cv2.INTER_LINEAR)
    if mask_path == "":
        mask = np.zeros_like(img, dtype=np.uint8)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (CFG.img_size[1], CFG.img_size[0]), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0).astype(np.uint8)

    img = np.expand_dims(img.astype("float32"), axis=-1)
    mask = np.expand_dims(mask.astype("float32"), axis=-1)
    if scale:
        img = (img - img.min()) / (img.max() - img.min())
        # img = (img - img.mean()) / img.std()
    assert img.shape == mask.shape, f"Image shape {img.shape} does not match mask shape {mask.shape}"
    return img, mask


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self,
                df,
                transforms=None):

        self.df           = df.reset_index(drop=True)
        self.transforms   = transforms
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.image_path[index]
        label_path = self.df.label_path[index]

        ## Load the image (RGB)
        img, mask = load_img(img_path, label_path, True)
        ## Apply Augmentations:
        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img  = data['image']
            mask = data['mask']
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.transpose(img, (2, 0, 1))

        mask = np.transpose(mask, (2, 0, 1))

        # if CFG.aux_head and self.label:
        #     labels = np.where(mask.sum((1, 2)) > 0, 1, 0)
        # else:
        #     labels = mask
        img = torch.tensor(img)
        mask = torch.tensor(mask)
        return torch.tensor(img), torch.tensor(mask), img_path


class TTADataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, tta_transforms):
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask, img_path = self.base_dataset[idx]

        image = image.permute(1, 2, 0).numpy()

        all_aug_images = []
        for t in self.tta_transforms:
            aug = t(image=image)['image']
            aug = torch.from_numpy(aug).permute(2, 0, 1).float()  # back to CHW
            all_aug_images.append(aug)

        return torch.stack(all_aug_images), mask, img_path
