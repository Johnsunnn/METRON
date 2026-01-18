import torch
import os
import shutil
import random
import numpy as np
import logging
import sys
_logging_configured = False
import configs


def setup_seed(seed=configs.RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename_prefix=configs.SAVE_MODEL_PATH):
    if not os.path.exists(filename_prefix):
        os.makedirs(filename_prefix)
    epoch = state.get('epoch', 'unknown')
    checkpoint_name = os.path.join(filename_prefix, configs.CHECKPOINT_NAME.format(epoch))
    torch.save(state, checkpoint_name)
    logging.info(f"Checkpoint saved: {checkpoint_name}")
    if is_best:
        best_model_path = os.path.join(filename_prefix, configs.BEST_MODEL_NAME)
        shutil.copyfile(checkpoint_name, best_model_path)
        logging.info(f"Best model saved: {best_model_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=configs.DEVICE):
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('best_metric', float('inf'))

    return start_epoch, best_metric


def setup_logging(log_file='training.log', level=logging.INFO, force_reconfigure=False):
    global _logging_configured

    root_logger = logging.getLogger()

    if root_logger.hasHandlers() and not force_reconfigure:
        if _logging_configured:
            return

    if force_reconfigure:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)
    _logging_configured = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0