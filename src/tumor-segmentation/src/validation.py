import gc
import numpy as np
import pandas as pd
import torch
import wandb
import logging
from tqdm import tqdm
from .losses import criterion
from .metrics import get_dice

logger = logging.getLogger(__name__)


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
