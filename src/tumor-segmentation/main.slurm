#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="tumor-segmentation"
#SBATCH --time=03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="(a100|h100)&(gpu32g|gpu40g|gpu80g)"
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_tumor_seg_%j.txt
#SBATCH --error=slurm_outputs/output_tumor_seg_%j.err
#SBATCH --mail-user=sveinung.myhre@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/$(whoami)/tumor-segmentation/cv-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.11

# Activate the environment
conda activate ${ENV_PATH}
pip install -r main_requirements.txt

echo "Downloaded dependencies:"
pip freeze


# Run the Python script
python main.py

# Deactivate the environment
conda deactivate

echo "Job finished"

# "sbatch ./main.slurm"  to run the job
