#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --output=output.log

echo "starting job"

module load python/3.11.5

source /venv/bin/activate
echo "starting python"
python /CC_Sweep/sweep.py --agent dqn --seeds 9999 --output 'data/dqn_sweep_results'

echo "finished"
