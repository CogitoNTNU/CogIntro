from torch.utils.data import DataLoader
from .config import CFG
from .dataset import BuildDataset, TTADataset


def prepare_loaders(df, data_transforms, fold, non_empty=False):
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
