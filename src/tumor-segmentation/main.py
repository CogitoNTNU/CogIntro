#!/usr/bin/env python3

import json
import os
from pathlib import Path
import wandb

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

    wandb.finish()

    print("Script ended successfully")


if __name__ == "__main__":
    main()
