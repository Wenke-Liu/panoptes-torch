#!/bin/bash
#SBATCH --job-name=PanoptesTorch
#SBATCH --partition=gpu8_long,gpu4_long #a100_long
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --begin=now
#SBATCH --time=4-00:00:00
#SBATCH --mem=256GB
#SBATCH --output=trn_%j.out
#SBATCH --error=trn_%j.error
#SBATCH --gres=gpu:v100:4

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
srun python train.py --config='../scripts/serous-svs.yaml'