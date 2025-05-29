from datetime import datetime

CHECKPOINTS_FOLDER = '/cluster/project7/ProsRegNet_CellCount/UNet/checkpoints/'

def format_best_checkpoint_name():
    now = datetime.now()
    checkpoint_file = CHECKPOINTS_FOLDER + f'checkpoints_{now.strftime("%d%m")}_{now.strftime("%H%M")}_best.pth'
    return checkpoint_file

def format_current_checkpoint_name():
    now = datetime.now()
    checkpoint_file = CHECKPOINTS_FOLDER + f'checkpoints_{now.strftime("%d%m")}_{now.strftime("%H%M")}_current.pth'
    return checkpoint_file

def get_checkpoint_name():
    now = datetime.now()
    checkpoint_file = CHECKPOINTS_FOLDER + f'checkpoints_{now.strftime("%d%m")}_{now.strftime("%H%M")}'
    return checkpoint_file