import torch


DATA_PATH = './dataset/BA_Modeling_all_samples_raw_data.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
PRINT_FREQ = 5
SAVE_MODEL_PATH = './checkpoints/'
BEST_MODEL_NAME = 'best_model.pth'
CHECKPOINT_NAME = 'checkpoint_epoch_{}.pth'
metric_to_monitor = 'rmse'
METRIC_EPSILON = 1e-8
EVALUATE = False
dropout = 0.
droprate = 0.
RANDOM_SEED = 1014

def get_serializable_configs():
    return {
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'METRIC_EPSILON': METRIC_EPSILON,
        'RANDOM_SEED': RANDOM_SEED,
        'DROPOUT': dropout,
        'DROPRATE': droprate,
    }