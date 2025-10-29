import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
from .config import CFG
from .utils import c_, sr_


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
