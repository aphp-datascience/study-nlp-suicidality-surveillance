#!/bin/bash
#SBATCH --job-name=ent_classification_ml
#SBATCH -t 3:30:00
#SBATCH --gres=gpu:t4:2
#SBATCH -N1-1
#SBATCH -c2
#SBATCH --mem=8000
#SBATCH -p gpuT4
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/cse210013:/export/home/cse210013 --container-mount-home --container-writable   --container-workdir=/
#SBATCH --wait
#SBATCH --output=./logs/slurm/output-%j.out
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh

## your code here
NAME_CONF_FILE=$1
conda activate env_cse_210013

python $HOME/cse_210013/suicide_attempt/pipelines/ent_classification_ml.py --conf-name $NAME_CONF_FILE --device cuda --path-model ~/cse_210013/data/models/BertTokenEDS.ckpt