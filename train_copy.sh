#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/UNet
#$ -N Debug

date

echo "NVIDIA SMI output:"
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"


set -x

python3 train_debug.py --img_size 64 --save_as='default_64' --n_epochs 5

date