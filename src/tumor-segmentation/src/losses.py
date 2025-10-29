import segmentation_models_pytorch as smp


# Loss functions
#JaccardLoss    = smp.losses.JaccardLoss(mode='binary')
DiceLoss       = smp.losses.DiceLoss(mode='binary')
#BCELoss        = smp.losses.SoftBCEWithLogitsLoss()
#LovaszLoss     = smp.losses.LovaszLoss(mode='binary', per_image=False)
#TverskyLoss    = smp.losses.TverskyLoss(mode='binary', log_loss=False, smooth=0.1)
#SegFocalLoss   = smp.losses.FocalLoss(mode = 'binary')
#BCE = torch.nn.BCEWithLogitsLoss()


def criterion(y_pred, y_true):
    return DiceLoss(y_pred, y_true)
