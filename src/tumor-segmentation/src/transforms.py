import albumentations as A
from .config import CFG


# Data transforms configuration
data_transforms = {"train": A.Compose([A.HorizontalFlip(p=0.5),
                                   A.VerticalFlip(p=0.5),
                               #    A.ShiftScaleRotate(rotate_limit=25, scale_limit=0.15, shift_limit=0, p=0.25),
#                                        A.CoarseDropout(max_holes=16, max_height=64 ,max_width=64 ,p=0.5),
#                                        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, p=0.75),
#                                        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, p=0.5),
                                   A.RandomCrop(height=CFG.img_size[0], width=CFG.img_size[1], always_apply=True, p=1)
                                    ]),

                "valid": A.Compose([]),#PadToDivisible(divisible=32, always_apply=True, p=1.0),

                "tta": [
                    A.Compose([]),  # identity
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0)
                 ]
                }
