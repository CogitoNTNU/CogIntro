import os
import time
import gc
import copy
import numpy as np
import torch
import wandb
import logging
from tqdm import tqdm
from collections import defaultdict
from .config import CFG
from .losses import criterion
from .metrics import dice_coef, iou_coef

logger = logging.getLogger(__name__)


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


def run_training(model, optimizer, scheduler, num_epochs, train_loader, valid_loader, fold=0, run=None):
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

        # Import here to avoid circular import
        from .validation import valid_one_epoch

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
            if run is not None:
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
