#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/UNet
#$ -N Train_T2W_UNet

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

cd ../CriDiff
source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"
cd ../UNet

python3 train.py --load_checkpoint 'checkpoints_0306_1947'

date