#!/bin/bash

#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:2
#SBATCH --job-name=aharris6-job-3927382
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --output=/juice/scr/aharris6/output.out
#SBATCH --partition=jag-standard
#SBATCH --time=10-0

# activate your desired anaconda environment
source activate sasha

# cd to working directory
cd .

# launch commands
srun --unbuffered run_as_child_processes 'python3 train.py --checkpoint --train-manifest ../manifests_slurm/coraal_train_manifest.csv  --val-manifest ../manifests_slurm/coraal_val_manifest.csv --batch-size 10  —hidden—dim 548 epochs 2 --cuda --id coraal_hidden_548_lr1e-3'
