#!/bin/bash
#SBATCH --job-name=qrchyperparam
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=output.log

echo "starting job"

module load python/3.11.5
module load cuda/12.6

source venv/bin/activate
echo "starting python"
python CC_Sweep/sweep.py --agent qrc --seeds 9999 --jobs 12 --output 'data/qrc_sweep_results'

echo "finished"
