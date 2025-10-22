# Setup

0. Get access and login into Idun via the CLI

1. Download dataset into /data folder

1. Sign up to Weights & Biases and confiugure user (get added to the Cogintro team)

1. Retrive W&B api key (https://wandb.ai/authorize)

1. Rename the file `.env.example` to `.env` and add the api key to it.

1. Run a job on Idun:

```bash
sbatch main.slurm
```

6. Inspect program via W&B dashboard and logs:

```bash
cat slurm_outputs/output_tumor_seg.txt
```
