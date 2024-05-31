#!/bin/bash
#SBATCH -A research
#SBATCH -c 25
#SBATCH -w gnode068
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=2048
#SBATCH --output=cyws_resnext50_coam_3x_affine.out
#SBATCH -J CYWS_resnext50_COAM_3x_affine

module load u18/cuda/10.2 u18/cudnn/7.6.5-cuda-10.2

# activate anaconda environment
source activate cyws
echo "conda environment activated"

# run the main code
cd /home2/hardik.mittal/The-Change-You-Want-to-See
pwd
python3 main.py --method centernet --gpus 4 --config_file configs/detection_resnext50_3x_coam_layers_affine.yml --max_epochs 100 --decoder_attention scse --experiment_name detection_resnext50_3x_coam_layers_affine_100


# https://slurm.schedmd.com/sbatch.html