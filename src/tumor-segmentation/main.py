#!/usr/bin/env python3

import os
import gc
import wandb
import logging

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
import torch.optim as optim
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Loads the variables from the .env file into the runtime environment
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tumor_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.config import CFG
from src.utils import set_seed
from src.transforms import data_transforms
from src.model import build_model
from src.dataloader import prepare_loaders
from src.train import run_training
from src.validation import oof_one_epoch
from src.scheduler import fetch_scheduler


def main():

    logger.info("="*50)
    logger.info("Starting tumor segmentation training script")
    logger.info("="*50)

    # Load environment variables (from the .env file)
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")

    if not WANDB_API_KEY:
        logger.warning("WANDB_API_KEY not found in environment variables")
    else:
        logger.info("WANDB_API_KEY loaded successfully")

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

    set_seed(CFG.seed)


    logger.info("Starting data loading and preprocessing...")
    CONTROLS_DIR = "data/nmai/controls"
    PATIENTS_DIR = "data/nmai/patients"

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


    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold,(train_idx, val_idx) in enumerate(skf.split(df, df['label'], df['image_id'])):
        df.loc[val_idx, 'fold'] = fold
    fold_distribution = df.groupby(['fold','label'])['image_id'].count()
    logger.info(f"Cross-validation fold distribution:\n{fold_distribution}")


    train_loader, valid_loader, oof_loader, _, valid_df = prepare_loaders(df, data_transforms, fold=0)
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
    oof_dice_scores = []
    all_oof_dfs = []
    tta_transform_names = ["identity", "hflip", "vflip"]

    for fold in CFG.folds:
        logger.info(f'\n{"#"*30}\n##### Starting fold {fold} training \n{"#"*30}\n')
        run.name = f"fold{fold}_{datetime.now():%Y%m%d_%H%M%S}"
        logger.info(f"W&B run name updated to: {run.name}")

        model = build_model()

        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.05)

        # Loaders for this fold (train + valid + TTA OOF)
        logger.info(f"Preparing data loaders for fold {fold}...")
        train_loader, valid_loader, oof_loader, train_steps, valid_df = prepare_loaders(
            df, data_transforms, fold=fold, non_empty=False
        )

        scheduler = fetch_scheduler(optimizer, train_steps)

        # Train model
        logger.info(f"Starting training for fold {fold}...")
        model, _ = run_training(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=CFG.epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            fold=fold,
            run=run
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

        oof_dice_scores.append(oof_dice)
        all_oof_dfs.append(valid_df_with_scores)

    # Final average OOF Dice
    mean_oof_dice = np.mean(oof_dice_scores)
    std_oof_dice = np.std(oof_dice_scores)
    logger.info(f"Cross-validation completed - Mean OOF Dice: {mean_oof_dice:.4f} ± {std_oof_dice:.4f}")
    logger.info(f"Individual fold scores: {[f'{score:.4f}' for score in oof_dice_scores]}")
    print(f"\n{'='*40}\nFinal OOF Dice across all folds: {mean_oof_dice:.4f}")

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
