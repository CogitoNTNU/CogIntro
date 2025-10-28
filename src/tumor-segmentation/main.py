#!/usr/bin/env python3

import os
from pathlib import Path
import wandb
import logging

import numpy as np
import pandas as pd
# pd.options.plotting.backend = "plotly"
import random
from glob import glob
from tqdm import tqdm
tqdm.pandas()

from dataclasses import dataclass


# visualization
import cv2
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import StratifiedGroupKFold

# PyTorch
import segmentation_models_pytorch as smp

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# Albumentations for augmentations
import albumentations as A


# Add Monitoring and Logging
from datetime import datetime

# For colored terminal text
from colorama import Fore, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Loads the variables from the .env file into the runtime environment
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tumor_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CFG:
    seed          = 2025
    backbone      = "efficientnet-b3"# "efficientnet-b3" # se_resnext101_32x4d - se_resnext50_32x4d - efficientnet-b3  timm-resnest101e
    decoder_attention_type = "scse"
    train_bs      = 3
    valid_bs      = train_bs*2
    img_size      = [1536, 786]
    crop_size     = [512, 512]
    epochs        = 40# 40
    lr            = 1e-4
    max_grad_norm = 100
    scheduler     = "ReduceLROnPlateau" #'OneCycle' # ReduceLROnPlateau CosineAnnealingLR CustomCosineAnnealingWarmupRestarts
    min_lr        = 5e-5
    T_max         = int(17000/train_bs*epochs)+50
    T_0           = 25
    max_lr        = 4e-4
    warmup_epochs = 0
    wd            = 5e-6
    n_accumulate  = 1
    n_fold        = 4
    folds         = [0, 1, 2, 3]
    num_classes   = 1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    aux_head      = False
    thresh        = [0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    import os
    import time
    import gc
    import copy
    from collections import defaultdict




    # pd.options.plotting.backend = "plotly"
    from tqdm import tqdm
    tqdm.pandas()



    # visualization

    # Sklearn

    # PyTorch


    # Albumentations for augmentations


    # Add Monitoring and Logging

    # For colored terminal text
    from colorama import Fore, Style
    c_  = Fore.GREEN
    sr_ = Style.RESET_ALL


    logger.info("="*50)
    logger.info("Starting tumor segmentation training script")
    logger.info("="*50)

    # Load environment variables (from the .env file)
    logger.info("Loading environment variables...")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    #WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    
    if not WANDB_API_KEY:
        logger.warning("WANDB_API_KEY not found in environment variables")
    else:
        logger.info("WANDB_API_KEY loaded successfully")

    # Initialize W&B run
    logger.info("Initializing Weights & Biases...")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="tumor-segmentation",
        entity="cogintro",
        name=f"train_unet_kfold_tta_{datetime.now():%Y%m%d_%H%M%S}",
        config={
            "seed": CFG.seed,
            "backbone": CFG.backbone,
            "decoder_attention_type": CFG.decoder_attention_type,
            "train_bs": CFG.train_bs,
            "valid_bs": CFG.valid_bs,
            "img_size": CFG.img_size,
            "crop_size": CFG.crop_size,
            "epochs": CFG.epochs,
            "lr": CFG.lr,
            "max_grad_norm": CFG.max_grad_norm,
            "scheduler": CFG.scheduler,
            "min_lr": CFG.min_lr,
            "T_max": CFG.T_max,
            "T_0": CFG.T_0,
            "max_lr": CFG.max_lr,
            "warmup_epochs": CFG.warmup_epochs,
            "wd": CFG.wd,
            "n_accumulate": CFG.n_accumulate,
            "n_fold": CFG.n_fold,
            "folds": CFG.folds,
            "num_classes": CFG.num_classes,
            "aux_head": CFG.aux_head,
            "thresh": CFG.thresh,
        },
        tags=["segmentation", "efficientnet"],
        save_code=True,
    )
    logger.info(f"W&B run initialized: {run.name}")

    # Log patient image and segmentation
    logger.info("Logging sample patient image and segmentation to W&B...")
    patient_image_path = Path("data/autopet/patients/imgs/patient_000.png")
    segmentation_path = Path("data/autopet/patients/labels/segmentation_000.png")

    if patient_image_path.exists() and segmentation_path.exists():
        wandb.log(
            {
                "patient_image": wandb.Image(str(patient_image_path)),
                "segmentation": wandb.Image(str(segmentation_path)),
            }
        )
        logger.info("Sample images logged to W&B successfully")
    else:
        logger.warning(f"Sample images not found: {patient_image_path} or {segmentation_path}")





    


    logger.info(f"WANDB_API_KEY: {'*' * 10 if WANDB_API_KEY else 'Not set'}")


    def set_seed(seed = 42):
        '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
        logger.info(f"Setting random seed to {seed} for reproducibility...")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info('Random seed set successfully')

    set_seed(CFG.seed)



    logger.info("Starting data loading and preprocessing...")
    CONTROLS_DIR = "data/nmai/controls"
    PATIENTS_DIR = "data/nmai/patients"

    logger.info(f"Loading data from directories: {CONTROLS_DIR} and {PATIENTS_DIR}")
    rows = []
    data_df = pd.DataFrame(columns=['image_id', 'image_path', 'label_path', 'label'])

    # Load the control images
    logger.info("Loading control images...")
    control_images = glob(os.path.join(CONTROLS_DIR, "imgs", "*.png"))
    logger.info(f"Found {len(control_images)} control images")
    for img_path in control_images:
        image_id = os.path.basename(img_path).split('/')[-1]
        rows.append({'image_id': image_id, 'image_path': f'{CONTROLS_DIR}/imgs/{image_id}', 'label_path': '', 'label': 0})

    # Load the patient images
    logger.info("Loading patient images and segmentation labels...")
    patient_images = glob(os.path.join(PATIENTS_DIR, "imgs", "*.png"))
    segmentation_labels = glob(os.path.join(PATIENTS_DIR, "labels", "*.png"))
    logger.info(f"Found {len(patient_images)} patient images and {len(segmentation_labels)} segmentation labels")
    for img_path, _ in zip(patient_images, segmentation_labels):
        image_id = os.path.basename(img_path).split('/')[-1]
        label_id = image_id.replace('patient', 'segmentation')
        rows.append({'image_id': image_id, 'image_path': f'{PATIENTS_DIR}/imgs/{image_id}', 'label_path': f'{PATIENTS_DIR}/labels/{label_id}', 'label': 1})

    data_df = pd.DataFrame(rows)
    data_df = data_df.reset_index(drop=True)
    logger.info(f"Total dataset size: {len(data_df)} samples")
    


    df = data_df.copy()
    label_counts = df['label'].value_counts()
    logger.info(f"Dataset label distribution:\n{label_counts}")




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


    def get_dice(preds, masks, threshold=0.5, epsilon=1e-6):
        """
        Compute per-image Dice coefficient and return the mean across the batch.

        preds, masks: np.ndarray of shape (B, H, W) or (B, 1, H, W)
        """
        preds = (preds > threshold).astype(np.uint8)
        masks = (masks > threshold).astype(np.uint8)

        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds[:, 0]
            masks = masks[:, 0]

        intersection = (preds & masks).sum(axis=(1, 2))
        total = preds.sum(axis=(1, 2)) + masks.sum(axis=(1, 2))

        dice_scores = (2.0 * intersection + epsilon) / (total + epsilon)
        return dice_scores.mean()




    logger.info("Setting up cross-validation folds...")
    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold,(train_idx, val_idx) in enumerate(skf.split(df, df['label'], df['image_id'])):
        df.loc[val_idx, 'fold'] = fold
    fold_distribution = df.groupby(['fold','label'])['image_id'].count()
    logger.info(f"Cross-validation fold distribution:\n{fold_distribution}")
    print(f"Cross-validation fold distribution:\n{fold_distribution}")



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
    





    def prepare_loaders(fold, non_empty=False):
        train_df = df[df.fold != fold].reset_index(drop=True)
        valid_df = df[df.fold == fold].reset_index(drop=True)

        if non_empty:
            train_df = train_df[train_df['label'] == 0].reset_index(drop=True)
            valid_df = valid_df[valid_df['label'] == 0].reset_index(drop=True)

        train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
        valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

        # Wrap the validation dataset in a deterministic TTA wrapper
        base_oof_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])
        oof_dataset = TTADataset(base_oof_dataset, tta_transforms=data_transforms['tta'])

        train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs,
                                num_workers=8, shuffle=True, pin_memory=True, drop_last=False)

        valid_loader = DataLoader(valid_dataset, batch_size=1,
                                num_workers=8, shuffle=False, pin_memory=True)

        oof_loader = DataLoader(oof_dataset, batch_size=1,  # returns [1, T, C, H, W]
                                num_workers=8, shuffle=False, pin_memory=True)

        return train_loader, valid_loader, oof_loader, len(train_df) // CFG.train_bs, valid_df


    logger.info("Testing data loaders and visualizing sample data...")
    train_loader, valid_loader, oof_loader, _, valid_df = prepare_loaders(fold=0)
    logger.info(f"Data loaders prepared - Train: {len(train_loader)} batches, Valid: {len(valid_loader)} batches")
    
    imgs, masks, _ = next(iter(train_loader))
    logger.info(f"Sample batch shape - Images: {imgs.shape}, Masks: {masks.shape}")

    for i in range(imgs.shape[0]):
        #make the image and mask two suplots horizontally with third image overlaying them together
        _img = np.transpose(imgs[i].cpu().numpy(), (1, 2, 0))
        mask = np.transpose(masks[i].cpu().numpy(), (1, 2, 0))
        fig, ax = plt.subplots(1,3, figsize=(12, 6))
        ax[0].imshow(_img)
        ax[0].set_title("Image")
        ax[1].imshow(mask, alpha = 0.25)
        ax[1].set_title("Mask")
        ax[2].imshow(_img)
        ax[2].imshow(mask, alpha = 0.25)
        ax[2].set_title("Overlay")
        plt.show()

    imgs, msks, paths_ = next(iter(valid_loader))
    logger.info(f"Validation batch shape - Images: {imgs.shape}, Masks: {msks.shape}")

    for i in range(imgs.shape[0]):
        _img = np.transpose(imgs[i].cpu().numpy(), (1, 2, 0))
        _mask = np.transpose(msks[i].cpu().numpy(), (1, 2, 0))
        logger.info(f"Image {i} - Min: {_img.min():.4f}, Max: {_img.max():.4f}")
        logger.info(f"Mask {i} - Unique values: {np.unique(_mask)}")
        print(_img.max(), _img.min())
        print(np.unique(_mask))
        plt.imshow(_img)
        plt.imshow(_mask, alpha = 0.25, cmap = 'gray')
        plt.show()

    gc.collect()
    logger.info("Memory cleanup completed")




    def build_model():
        logger.info(f"Building U-Net model with backbone: {CFG.backbone}")
        model = smp.Unet(encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=1,                      # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_attention_type = CFG.decoder_attention_type, #"scse",
                        aux_params = None if not CFG.aux_head else {"classes": 1,
                                                                    "activation": None})
        model.to(CFG.device)
        logger.info(f"Model built and moved to device: {CFG.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model


    def load_model(path):
        logger.info(f"Loading model from: {path}")
        model = build_model()
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info("Model loaded and set to evaluation mode")
        return model



    #JaccardLoss    = smp.losses.JaccardLoss(mode='binary')
    DiceLoss       = smp.losses.DiceLoss(mode='binary')
    #BCELoss        = smp.losses.SoftBCEWithLogitsLoss()
    #LovaszLoss     = smp.losses.LovaszLoss(mode='binary', per_image=False)
    #TverskyLoss    = smp.losses.TverskyLoss(mode='binary', log_loss=False, smooth=0.1)
    #SegFocalLoss   = smp.losses.FocalLoss(mode = 'binary')
    #BCE = torch.nn.BCEWithLogitsLoss()

    def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=1e-6):
        y_true = y_true.float()
        y_pred = (y_pred > thr).float()
        inter = (y_true * y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = (2 * inter + epsilon) / (den + epsilon)
        return dice.mean()  # mean over batch (and channel if present)


    def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=1e-6):
        y_true = y_true.float()
        y_pred = (y_pred > thr).float()
        inter = (y_true * y_pred).sum(dim=dim)
        union = y_true.sum(dim=dim) + y_pred.sum(dim=dim) - inter
        iou = (inter + epsilon) / (union + epsilon)
        return iou.mean()  # mean over batch


    def criterion(y_pred, y_true):
        return DiceLoss(y_pred, y_true)

    # def criterion(y_pred, y_true):
    #     if CFG.aux_head:
    #         y_true, yt_class = y_true
    #         y_pred, yp_class = y_pred
    #         return (0.5*DiceLoss(y_pred, y_true) + 0.5 * BCE(yp_class, yt_class))
    #     return 0.5*DiceLoss(y_pred, y_true) + 0.5*SegFocalLoss(y_pred, y_true)


    def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch:int):
        logger.info(f"Starting training epoch {epoch}")
        model.train()

        dataset_size = 0
        running_loss = 0.0
        train_jaccards = []
        train_dices = []

        sigmoid = torch.sigmoid  # Faster than instantiating nn.Sigmoid()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')

        optimizer.zero_grad()

        for step, (images, masks, paths) in pbar:
            images = images.to(device, dtype=torch.float)
            masks  = masks.to(device, dtype=torch.float)
            batch_size = images.size(0)

            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss.backward()

            y_pred = sigmoid(y_pred)

            train_dice = dice_coef(masks, y_pred).cpu().item()
            train_jaccard = iou_coef(masks, y_pred).cpu().item()
            train_dices.append(train_dice)
            train_jaccards.append(train_jaccard)

            if (step + 1) % CFG.n_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

                if scheduler and CFG.scheduler not in ["ReduceLROnPlateau", "ExponentialLR"]:
                    scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size
            wandb.log({'train_loss': running_loss, 'epoch': epoch})
            # W&B per-epoch training metrics
            current_lr = optimizer.param_groups[0]['lr']
            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

            pbar.set_postfix(loss=f'{epoch_loss:0.4f}',
                            lr=f'{current_lr:0.5f}',
                            jac=np.mean(train_jaccards),
                            dice=np.mean(train_dices),
                            gpu_mem=f'{mem:0.2f} GB')

        torch.cuda.empty_cache()
        gc.collect()
        
        epoch_dice = np.mean(train_dices)
        epoch_jaccard = np.mean(train_jaccards)
        logger.info(f"Epoch {epoch} training completed - Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, Jaccard: {epoch_jaccard:.4f}")

        return epoch_loss, epoch_dice, epoch_jaccard


    @torch.no_grad()
    def valid_one_epoch(model, dataloader, device, optimizer, epoch:int):
        logger.info(f"Starting validation epoch {epoch}")
        model.eval()

        dataset_size = 0
        running_loss = 0.0
        global_masks = []
        global_preds = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')

        for step, (images, masks, paths) in pbar:
            images = images.float().to(device)
            masks = masks.float().to(device)
            batch_size = images.size(0)

            y_pred = model(images)
            loss = criterion(y_pred, masks)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            y_pred = torch.sigmoid(y_pred)
            global_masks.append(masks.cpu().numpy())
            global_preds.append(y_pred.detach().cpu().numpy())

            current_lr = optimizer.param_groups[0]['lr']
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                            lr=f'{current_lr:0.5f}',
                            gpu_mem=f'{mem:0.2f} GB')

        # For sample images, take first example of last batch
        # Concatenate all batches
        global_masks = np.concatenate(global_masks, axis=0)
        global_preds = np.concatenate(global_preds, axis=0)
        global_dice = get_dice(global_preds, global_masks)

        # Log overall validation metrics
        wandb.log({'val_loss': epoch_loss, 'val_dice': global_dice, 'epoch': epoch})

        # For sample images, take first example of last batch
        img_np = images[0].cpu().permute(1,2,0).numpy()
        mask_np = masks[0].cpu().permute(1,2,0).numpy()
        pred_np = global_preds[-1][0].astype('float32') # Remove transpose as it's a single channel
        wandb.log({
            'input': wandb.Image(img_np, caption='input'),
            'mask': wandb.Image(mask_np, caption='mask'),
            'pred': wandb.Image(pred_np, caption='pred'),
            'epoch': epoch
        })

        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Epoch {epoch} validation completed - Loss: {epoch_loss:.4f}, Dice: {global_dice:.4f}")

        return epoch_loss, global_dice

    

    def reverse_transform(pred, transform_type):
        if transform_type == "hflip":
            return np.fliplr(pred)
        elif transform_type == "vflip":
            return np.flipud(pred)
        elif transform_type == "identity":
            return pred
        else:
            raise ValueError(f"Unknown TTA transform: {transform_type}")

    @torch.no_grad()
    def oof_one_epoch(model, dataloader, device, valid_df, fold, tta_transform_names):
        logger.info(f"Starting out-of-fold evaluation for fold {fold}")
        model.eval()

        oof_scores = []
        global_preds = []
        global_masks = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='OOF Eval')

        for step, (tta_images, masks, img_path) in pbar:
            tta_images = tta_images.squeeze(0).to(device).float()
            masks = masks.squeeze(0).to(device).float()
            img_path = img_path[0] if isinstance(img_path, list) else img_path
            img_path = str(img_path)  # ensure string for merging

            preds = model(tta_images)
            preds = torch.sigmoid(preds).squeeze(1).cpu().numpy()
            masks_np = masks.squeeze().cpu().numpy()

            aligned_preds = []
            for pred, tname in zip(preds, tta_transform_names):
                aligned_preds.append(reverse_transform(pred, tname))

            aligned_preds = np.stack(aligned_preds, axis=0)
            tta_avg_pred = aligned_preds.mean(axis=0)
            base_pred = aligned_preds[0]

            base_dice = get_dice(base_pred[None], masks_np[None])
            tta_dice = get_dice(tta_avg_pred[None], masks_np[None])

            global_preds.append(tta_avg_pred[None])
            global_masks.append(masks_np[None])

            oof_scores.append({
                'image_path': img_path,
                'base_dice': base_dice,
                'tta_dice': tta_dice
            })

            pbar.set_postfix(base_dice=f'{base_dice:.4f}', tta_dice=f'{tta_dice:.4f}')

        df_scores = pd.DataFrame(oof_scores)
        logger.info(f"OOF evaluation completed for {len(df_scores)} samples")

        # Merge on image_path instead of index
        valid_df = valid_df.copy()
        valid_df = valid_df.merge(df_scores, on='image_path', how='left')
        valid_df.to_csv(f'tta_results_fold_{fold}.csv', index=False)
        logger.info(f"TTA results saved to tta_results_fold_{fold}.csv")

        global_preds = np.concatenate(global_preds, axis=0)
        global_masks = np.concatenate(global_masks, axis=0)
        global_dice = get_dice(global_preds, global_masks)
        logger.info(f"Global OOF Dice score: {global_dice:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

        return global_dice, valid_df




    import torch
    import gc
    import numpy as np
    import os

    def run_training(model, optimizer, scheduler, num_epochs, train_loader, valid_loader, fold=0):
        logger.info(f"Starting training for fold {fold}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA: {torch.cuda.get_device_name()}\n")

        # Create the 'models' directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        logger.info("Models directory created/verified")

        wandb.watch(model, log='all', log_freq=10)
        logger.info("W&B model watching enabled")

        start_time = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_dice = -np.inf
        best_epoch = -1
        history = defaultdict(list)
        logger.info(f"Training configuration - Epochs: {num_epochs}, Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

        for epoch in range(1, num_epochs + 1):
            gc.collect()
            logger.info(f"Starting epoch {epoch}/{num_epochs}")
            print(f"{'='*30}\nEpoch {epoch}/{num_epochs}")

            train_loss, train_dice, train_jaccard = train_one_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                device=CFG.device,
                epoch=epoch,
            )

            val_loss, val_dice = valid_one_epoch(
                model=model,
                dataloader=valid_loader,
                device=CFG.device,
                optimizer=optimizer,
                epoch=epoch
            )

            history['Train Loss'].append(train_loss)
            history['Valid Loss'].append(val_loss)
            history['Valid Dice'].append(val_dice)

            logger.info(f"Epoch {epoch} results - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train Jaccard: {train_jaccard:.4f}, Valid Loss: {val_loss:.4f}, Valid Dice: {val_dice:.4f}")
            print(f"Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f} - Train Jaccard: {train_jaccard:.4f} | Valid Loss: {val_loss:.4f} | Valid Dice: {val_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                logger.info(f"New best model found! Dice improved from {best_dice:.4f} to {val_dice:.4f}")
                print(f"✓ Dice Improved: {best_dice:.4f} → {val_dice:.4f}")
                best_dice = val_dice
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

                best_path = f'models/best_fold{fold}_dice{best_dice:.4f}.pth'
                torch.save(model.state_dict(), best_path)
                logger.info(f"Best model saved to {best_path}")
                print(f"✔ Model saved to {best_path}")
                # W&B artifact
                artifact = wandb.Artifact(f'best_model_fold{fold}', type='model')
                artifact.add_file(best_path)
                run.log_artifact(artifact)
                logger.info("Model artifact logged to W&B")

            # Always save last epoch
            last_path = f"last_epoch-S1-{fold:02d}.bin"
            torch.save(model.state_dict(), last_path)

            # Step scheduler if applicable
            if CFG.scheduler in ["ReduceLROnPlateau", "ExponentialLR"]:
                if CFG.scheduler == "ExponentialLR":
                    scheduler.step()
                elif CFG.scheduler == "ReduceLROnPlateau":
                    scheduler.step(val_loss)

            print()

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        logger.info(f"Training completed in {h}h {m}m {s}s")
        logger.info(f"Best Dice score: {best_dice:.4f} achieved at epoch {best_epoch}")
        print(f"🏁 Training complete in {h}h {m}m {s}s")
        print(f"🏆 Best Dice: {best_dice:.4f} (Epoch {best_epoch})")

        model.load_state_dict(best_model_wts)
        logger.info("Best model weights loaded for inference")
        return model, history



    def fetch_scheduler(optimizer: Optimizer, training_steps: int = 0):
        match CFG.scheduler:
            case "CosineAnnealingLR":
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=training_steps * CFG.epochs, eta_min=CFG.min_lr
                )
            case "CosineAnnealingWarmRestarts":
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=training_steps * 8, T_mult=1, eta_min=CFG.min_lr
                )
            case "ReduceLROnPlateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=1,
                    cooldown=1,
                    min_lr=5e-6,
                    threshold=0.00001,
                )
            case "OneCycle":
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=CFG.max_lr,
                    total_steps=training_steps * CFG.epochs,
                    # epochs=CFG.epochs,
                    # steps_per_epoch=training_steps,
                    pct_start=0.25,
                )
            case "ExponentialLR":
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            case _:
                print(
                    f"{c_}⚠️ WARNING: Unknown scheduler {CFG.scheduler}. Using StepLR with step_size=1.{sr_}"
                )
                return None

        return scheduler


    logger.info("Starting k-fold cross-validation training...")
    oof_dice_scores = []
    all_oof_dfs = []
    tta_transform_names = ["identity", "hflip", "vflip"]

    for fold in CFG.folds:
        logger.info(f"Starting fold {fold} training")
        print(f'\n{"#"*30}\n##### Fold {fold}\n{"#"*30}\n')
        run.name = f"fold{fold}_{datetime.now():%Y%m%d_%H%M%S}"
        logger.info(f"W&B run name updated to: {run.name}")
        
        model = build_model()

        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.05)
        CFG.scheduler = "CosineAnnealingLR"
        logger.info(f"Optimizer initialized with lr={CFG.lr}, weight_decay=0.05")

        # Loaders for this fold (train + valid + TTA OOF)
        logger.info(f"Preparing data loaders for fold {fold}...")
        train_loader, valid_loader, oof_loader, train_steps, valid_df = prepare_loaders(
            fold=fold,
            non_empty=False,
        )

        scheduler = fetch_scheduler(optimizer, train_steps)
        logger.info(f"Scheduler initialized: {CFG.scheduler}")

        # Train model
        logger.info(f"Starting training for fold {fold}...")
        model, _ = run_training(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=CFG.epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            fold=fold
        )

        # TTA-based OOF prediction
        logger.info(f"Starting TTA-based out-of-fold evaluation for fold {fold}...")
        oof_dice, valid_df_with_scores = oof_one_epoch(
            model=model,
            dataloader=oof_loader,
            device=CFG.device,
            valid_df=valid_df,
            fold=fold,
            tta_transform_names=tta_transform_names
        )

        logger.info(f"Fold {fold} completed - OOF Dice: {oof_dice:.4f}")
        print(f"✅ Fold {fold} OOF Dice: {oof_dice:.4f}")

        oof_dice_scores.append(oof_dice)
        all_oof_dfs.append(valid_df_with_scores)

    # Final average OOF Dice
    mean_oof_dice = np.mean(oof_dice_scores)
    std_oof_dice = np.std(oof_dice_scores)
    logger.info(f"Cross-validation completed - Mean OOF Dice: {mean_oof_dice:.4f} ± {std_oof_dice:.4f}")
    logger.info(f"Individual fold scores: {[f'{score:.4f}' for score in oof_dice_scores]}")
    print(f"\n{'='*40}\n🏁 Final OOF Dice across all folds: {mean_oof_dice:.4f}")

    # Save full OOF dataframe
    logger.info("Saving out-of-fold results...")
    final_oof_df = pd.concat(all_oof_dfs, ignore_index=True)
    final_oof_df.to_csv("oof_scores_all_folds.csv", index=False)
    logger.info(f"OOF results saved to oof_scores_all_folds.csv ({len(final_oof_df)} samples)")

    # Finish W&B
    logger.info("Finalizing W&B run...")
    run.finish()
    logger.info("W&B run finished successfully")

    logger.info("="*50)
    logger.info("Tumor segmentation training script completed successfully")
    logger.info("="*50)
    print("Script ended successfully")


if __name__ == "__main__":
    main()
