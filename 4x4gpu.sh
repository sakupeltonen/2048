#!/bin/bash
#SBATCH --job-name=myTest
#SBATCH --account=project_2008407
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G

source ./myenv/bin/activate
srun python /projappl/project_2008407/senpai/main.py --cuda --agent-name 'DQN-4x4' --save-model