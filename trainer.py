import os
import torch
from tqdm import tqdm
import time
import logging
import configs
from utils import AverageMeter, save_checkpoint
from metrics import get_metrics


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                 device=configs.DEVICE, current_fold=None, checkpoint_dir=configs.SAVE_MODEL_PATH):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_metric = {'ang_diff': float('inf'), 'wsatl': float('inf'), 'mae': float('inf'),
                                'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf'), 'mape': float('inf')}
        self.metric_to_monitor = configs.metric_to_monitor
        self.start_epoch = 0
        self.current_fold = current_fold
        self.checkpoint_dir = checkpoint_dir

    def _train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{configs.NUM_EPOCHS} [Train]", leave=False)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.float()
            outputs = self.model(inputs)
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), inputs.size(0))

            if (i + 1) % configs.PRINT_FREQ == 0 or (i + 1) == len(self.train_loader):
                progress_bar.set_postfix({'Loss': f'{losses.avg:.4f}'})
            logging.info(
                f"Epoch [{epoch + 1}/{configs.NUM_EPOCHS}] Training Loss ({self.criterion.__class__.__name__}): {losses.avg:.4f}")

        return losses.avg

    def _process_epoch_results(self, all_predictions, all_targets, epoch_loss, epoch_type="Validation"):
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        metrics = get_metrics(predictions_tensor, targets_tensor)
        log_msg = f"Epoch [{self.current_epoch + 1}/{configs.NUM_EPOCHS}] {epoch_type} Results: "
        log_msg += f"Loss ({self.criterion.__class__.__name__}): {epoch_loss:.4f}, "
        for name, value in metrics.items():
            log_msg += f"{name.upper()}: {value:.4f}, "
        logging.info(log_msg.strip(", "))

        return metrics

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        self.current_epoch = epoch
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{configs.NUM_EPOCHS} [Validate]", leave=False)

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                progress_bar.set_postfix({'Val Loss': f'{losses.avg:.4f}'})
        val_metrics = self._process_epoch_results(all_predictions, all_targets, losses.avg, "Validation")

        return val_metrics

    def train(self, start_epoch=0, num_epochs=configs.NUM_EPOCHS):
        self.start_epoch = start_epoch
        fold_info = f"[Fold {self.current_fold + 1}/{configs.K_FOLDS}] " if self.current_fold is not None else ""
        logging.info(f"Starting training from epoch {self.start_epoch + 1} for {num_epochs - self.start_epoch} epochs.")

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            _ = self._train_one_epoch(epoch)
            val_metrics = self._validate_one_epoch(epoch)
            current_metric_val = val_metrics

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric_val[self.metric_to_monitor])
                else:
                    self.scheduler.step()

            epoch_duration = time.time() - epoch_start_time
            logging.info(f"{fold_info}Epoch [{epoch+1}/{num_epochs}] completed in "
                         f"{epoch_duration:.2f}s. Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            is_best = current_metric_val[self.metric_to_monitor] < self.best_val_metric[self.metric_to_monitor]
            if is_best:
                self.best_val_metric = current_metric_val
                logging.info(f"New best model found at epoch {epoch+1} with Val metrics : {self.best_val_metric}")
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_metric': self.best_val_metric,
                    'metric_monitored': self.metric_to_monitor,
                    'configs': configs.get_serializable_configs()
                }, is_best, filename_prefix=self.checkpoint_dir)

        logging.info(f"{fold_info}Training finished.")
        logging.info(f"{fold_info}Best validation metric: {self.best_val_metric}")

        return self.best_val_metric

    def evaluate(self, data_loader):
        self.model.eval()
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        progress_bar = tqdm(data_loader, desc="[Test]", leave=False)
        self.current_epoch = -1

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                progress_bar.set_postfix({'Loss': f'{losses.avg:.4f}'})

        eval_metrics = self._process_epoch_results(all_predictions, all_targets, losses.avg, "test set")
        all_predictions = [item.item() for tensor in all_predictions for item in tensor]

        return eval_metrics, all_predictions