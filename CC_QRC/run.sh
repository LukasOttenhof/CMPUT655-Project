#!/bin/bash
#SBATCH --job-name=train-dqn-epsilon
#SBATCH --account=def-cepp
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --output=output.log
#SBATCH --mail-user=rany@ualberta.ca
#SBATCH --mail-type=END

echo "starting job"

nvidia-smi

module load python/3.11.5
module load cuda/12.6
source cc/bin/activate
python CC_QRC/qrc.py

echo "finished"
