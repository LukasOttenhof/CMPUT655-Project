#!/bin/bash
#SBATCH --job-name=train-large-bs4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=output.log
#SBATCH --mail-user=rany@ualberta.ca
#SBATCH --mail-type=END

echo "starting job"

nvidia-smi

module load python/3.11
module load cuda/12.6
virtualenv --no-download $SLURM_TMPDIR/cc
source $SLURM_TMPDIR/cc/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python3 CC_QRC/qrc.py

echo "finished"
