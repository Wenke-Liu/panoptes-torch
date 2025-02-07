#!/bin/bash
#SBATCH --job-name=PanoptesTorch
#SBATCH --partition=gpu8_medium,gpu4_medium #a100_long
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --begin=now
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=tst_%j.out
#SBATCH --error=tst_%j.error
#SBATCH --gres=gpu:v100:1

#module load condaenvs/gpu/tensorflow2.2
module load cuda/11.8
source ~/.bashrc
#conda activate wsi
conda activate pt2
cd /gpfs/scratch/liuw09/panoptes_torch/src
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
srun python test.py --config='../scripts/serous-svs.yaml'