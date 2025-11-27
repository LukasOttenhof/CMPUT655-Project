#!/bin/bash
#SBATCH --job-name=train-large-bs4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=output.log

echo "starting job"

module load python/3.11.5

source /home/lottenho/CC_QRC/virtual_env/bin/activate
echo "starting python"
python /home/lottenho/CC_QRC/qrc.py

echo "finished"
