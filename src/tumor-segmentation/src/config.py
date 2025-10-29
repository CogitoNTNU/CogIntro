import torch
from dataclasses import dataclass


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
    scheduler     = "CosineAnnealingLR" #'OneCycle' # ReduceLROnPlateau CosineAnnealingLR CustomCosineAnnealingWarmupRestarts
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
