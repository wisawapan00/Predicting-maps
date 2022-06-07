import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


test = A.Compose(
    [   
        A.Resize(width=256, height=256),
        ToTensorV2(),
    ],  additional_targets={"image0":"image"}
)

train_augmentation = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],  additional_targets={"image0":"image"}
)

val_augmentation = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],  additional_targets={"image0":"image"}
)