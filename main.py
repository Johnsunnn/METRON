import torch.nn as nn
import torch.optim as optim
import os
import argparse
import configs
from data_loader import get_ba_dataset
from model import get_model
from trainer import Trainer
from utils import setup_seed, setup_logging, load_checkpoint
import joblib
import numpy as np
import pandas as pd
import torch
import logging
from metrics import get_metrics


def get_age_performance(predictions):
    predictions = np.array(predictions).reshape(-1, 1)
    label_scaler = joblib.load('./dataset/label_scaler.pkl')
    restored_predictions = label_scaler.inverse_transform(predictions)
    ground_truth = pd.read_csv('./checkpoints/combine_result.csv')
    ground_truth = ground_truth[ground_truth['Group'] == 'Validation']
    ca = ground_truth['CA'].values
    ca = torch.tensor(ca)
    ba = torch.tensor(restored_predictions)
    ba = ba.squeeze()
    metrics = get_metrics(ba, ca)
    for name, value in metrics.items():
        logging.info(f"{name.upper()}: {value}")

    return metrics


def run_train_eval(args, train_loader, val_loader, test_loader_for_final_eval):
    model = get_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE, weight_decay=configs.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    start_epoch = 0
    current_checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(current_checkpoint_dir):
        os.makedirs(current_checkpoint_dir)

    trainer_instance = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler,
                               configs.DEVICE, checkpoint_dir=current_checkpoint_dir)

    fold_best_metrics = trainer_instance.train(start_epoch=start_epoch, num_epochs=configs.NUM_EPOCHS)

    best_model_path_this_run = os.path.join(current_checkpoint_dir, configs.BEST_MODEL_NAME)
    load_checkpoint(best_model_path_this_run, model, device=configs.DEVICE)

    logging.info(f"Evaluating best model on the separate test set...")
    test_metrics = trainer_instance.evaluate(test_loader_for_final_eval)
    logging.info(f"Test Set metrics:\n {test_metrics}")

    return model


def main(args):
    setup_seed(configs.RANDOM_SEED)
    base_log_filename = 'training.log' if not args.evaluate else 'evaluation.log'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    main_log_file = os.path.join(args.checkpoint_dir, base_log_filename)
    setup_logging(main_log_file)
    logging.info(f"Using device: {configs.DEVICE}")
    logging.info(f"Batch size: {configs.BATCH_SIZE}, Epochs: {configs.NUM_EPOCHS}, LR: {configs.LEARNING_RATE}")
    _, true_test_loader = get_ba_dataset(configs.BATCH_SIZE)
    if args.evaluate:
        best_model_path = os.path.join(args.checkpoint_dir, configs.BEST_MODEL_NAME)
        model_to_eval = get_model()
        load_checkpoint(best_model_path, model_to_eval, device=configs.DEVICE)
        eval_trainer = Trainer(model_to_eval, None, None, nn.MSELoss(), None, device=configs.DEVICE)
        test_metrics, results = eval_trainer.evaluate(true_test_loader)
        get_age_performance(results)
        return

    train_loader, val_loader = get_ba_dataset(configs.BATCH_SIZE)
    trained_model_single = run_train_eval(args, train_loader, val_loader, true_test_loader)
    best_model_path_single = os.path.join(args.checkpoint_dir, configs.BEST_MODEL_NAME)
    model_for_test = get_model()
    load_checkpoint(best_model_path_single, model_for_test, device=configs.DEVICE)
    test_eval_trainer = Trainer(model_for_test, None, None, nn.MSELoss(), None, device=configs.DEVICE)
    test_metrics, results = test_eval_trainer.evaluate(true_test_loader)
    get_age_performance(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='METRON')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--evaluate', default=configs.EVALUATE, type=bool)
    parser.add_argument('--checkpoint-dir', default=configs.SAVE_MODEL_PATH, type=str)
    args = parser.parse_args()
    main(args)