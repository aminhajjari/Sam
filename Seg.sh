#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --time=02:00:00 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 
#SBATCH --job-name=MILK10k_Seg 

# Output and error files will be placed next to the Python script.
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2/MILK10k_Seg-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2/MILK10k_Seg-%j.err

echo "Starting job on $(hostname) at $(date)"

# Load the required modules
module load python/3.11.5
module load scipy-stack
module load cuda/12.6
module load opencv/4.12.0
echo "Modules loaded."

# Activate your virtual environment
source /lustre06/project/def-arashmoh/shahab33/XAI/milk10k_env/bin/activate
echo "Virtual environment activated."

# Navigate to the working directory
cd /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input
echo "Working directory set to $(pwd)"

# Run your Python script
python segment-anything-2/Seg.py

echo "Job finished at $(date)"