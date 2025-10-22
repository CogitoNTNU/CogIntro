#!/usr/bin/env python3

import json
import os
from pathlib import Path
import wandb
import nibabel as nib

# Loads the variables from the .env file into the runtime environment
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Script started")

    dataset_path = Path("data/psma-fdg-pet-ct-lesions/dataset.json")
    with open(dataset_path, "r") as f:
        dataset_info = json.load(f)

    print(f"Loaded dataset: {dataset_info['name']}")

    # Load environment variables (from the .env file)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_username = os.getenv("WANDB_ENTITY")

    wandb.login(key=wandb_api_key)

    wandb.init(project="tumor-segmentation", entity=wandb_username, config=dataset_info)

    wandb.log(
        {
            "dataset_name": dataset_info["name"],
            "num_training_samples": dataset_info["numTraining"],
            "num_labels": len(dataset_info["labels"]),
        }
    )

    # Load and display one of the images in the W&B dashboard
    image_path = Path(
        "data/psma-fdg-pet-ct-lesions/imagesTr/psma_ffcaa75377465b37_2018-03-04_0001.nii.gz"
    )
    nifti_image = nib.load(image_path)
    image_data = nifti_image.get_fdata()

    print(f"Loaded image: {image_path.name}, shape: {image_data.shape}")

    # Log a few slices
    middle_slice = image_data[:, :, image_data.shape[2] // 2 + 40]
    wandb.log({"sample_image1": wandb.Image(middle_slice, caption=image_path.name)})

    middle_slice = image_data[:, :, image_data.shape[2] // 2 + 50]
    wandb.log({"sample_image2": wandb.Image(middle_slice, caption=image_path.name)})

    middle_slice = image_data[:, :, image_data.shape[2] // 2 + 60]
    wandb.log({"sample_image3": wandb.Image(middle_slice, caption=image_path.name)})

    wandb.finish()

    print("Script ended successfully")


if __name__ == "__main__":
    main()
