#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=40:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/UNet
#$ -N Test_UNet

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

cd ../CriDiff
source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"
cd ../UNet

python3 test.py --img_size 128 --checkpoint 'checkpoints_0206_1629_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_0206_1629_stage_2_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_0406_1632_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_0406_1632_stage_2_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_0606_1731_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_2905_1924_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_2905_2111_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_2905_2111_stage_2_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_3005_0921_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_3005_1030_stage_1_best'
python3 test.py --img_size 128 --checkpoint 'checkpoints_3005_1030_stage_2_best'

date