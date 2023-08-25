#!/bin/bash
#SBATCH --job-name=train_camembert
#SBATCH -t 3:30:00
#SBATCH --gres=gpu:t4:2
#SBATCH -N1-1
#SBATCH -c2
#SBATCH --mem=4000
#SBATCH -p gpuT4
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER --container-mount-home --container-writable   --container-workdir=/
#SBATCH --output=./logs/slurm/output-%j.out
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh

conda activate env_cse_210013
NAME=$1
EPOCHS=$2
LR=$3
BATCH_SIZE=$4

python $HOME/cse_210013/suicide_attempt/pipelines/train_ml_model.py --max-epochs $EPOCHS --name $NAME --no-hp-search --lr $LR --batch-size $BATCH_SIZE