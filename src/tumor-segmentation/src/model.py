import torch
import segmentation_models_pytorch as smp
import logging
from .config import CFG

logger = logging.getLogger(__name__)


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
