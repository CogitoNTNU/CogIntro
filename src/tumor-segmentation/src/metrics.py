import numpy as np


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
