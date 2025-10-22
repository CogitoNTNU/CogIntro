#!/usr/bin/env python3

import os
from pathlib import Path
import wandb

# Loads the variables from the .env file into the runtime environment
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Script started")

    # Load environment variables (from the .env file)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_username = os.getenv("WANDB_ENTITY")

    wandb.login(key=wandb_api_key)

    wandb.init(project="tumor-segmentation", entity=wandb_username)

    # Log patient image and segmentation
    patient_image_path = Path("data/autopet/patients/imgs/patient_000.png")
    segmentation_path = Path("data/autopet/patients/labels/segmentation_000.png")

    wandb.log(
        {
            "patient_image": wandb.Image(str(patient_image_path)),
            "segmentation": wandb.Image(str(segmentation_path)),
        }
    )

    wandb.finish()

    print("Script ended successfully")


if __name__ == "__main__":
    main()
