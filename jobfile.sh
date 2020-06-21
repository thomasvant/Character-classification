#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --mincpus=8
#SBATCH --mem=10000
#SBATCH --workdir=/home/nfs/tvantussenbroe/NLP_project/Character-classification
#SBATCH --job-name=elmo
#SBATCH --output=/home/nfs/tvantussenbroe/NLP_project/Character-classification/output.txt
#SBATCH --error=/home/nfs/tvantussenbroe/NLP_project/Character-classification/errors.txt
#SBATCH --gres=gpu:0
module use /opt/insy/modulefiles
module load cuda/10.1 cudnn/10.1-7.6.0.64
source ~/NLP_project/Character-classification/venv/bin/activate
echo "Starting at $(date)"
srun python main.py
echo "Finished at $(date)"