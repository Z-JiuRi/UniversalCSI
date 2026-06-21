import time
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from utils import logger
from utils.statics import AverageMeter, evaluator, nmse_from_sums

__all__ = ['Trainer', 'Tester']

class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', tensorboard_dir=None, print_freq=20,
                 val_freq=10, test_freq=10, test_every_epoch=False):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.tensorboard_dir = tensorboard_dir
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq
        self.test_every_epoch = test_every_epoch

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_nmse = {'nmse': None, 'epoch': None}
        self.last_train_metrics = {}
        self.last_val_metrics = {}
        self.last_test_metrics = {}

        self.tester = Tester(model, device, criterion, print_freq)
        self.test_loader = None
        if self.tensorboard_dir is None:
            self.tensorboard_dir = os.path.join("exps", "default", "tensorboard")
        self.vision = SummaryWriter(log_dir=self.tensorboard_dir)

    def loop(self, epochs, train_loader, val_loader, test_loader):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
        """

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            self.train_loss = self.train(train_loader)
            if ep % self.val_freq == 0:
                self.val_loss = self.val(val_loader)

            if self.test_every_epoch or ep % self.test_freq == 0:
                self.test_loss, nmse = self.test(test_loader)
                self.vision.add_scalar("test/loss", self.test_loss, global_step=ep)
                self.vision.add_scalar("test/nmse", nmse, global_step=ep)
                self.vision.add_scalar("test/train_loss", self.train_loss, global_step=ep)
                self._write_encoder_scalars(ep)
            else:
                nmse = None

            # conduct saving, visualization and log printing
            self._write_adapter_scalars(ep)
            self._loop_postprocessing(nmse)

    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        r""" exam the model with validation set.

        Args:
            val_loader: (DataLoader): the validation data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader):
        r""" Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        encoder_losses = {}
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0]
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            if isinstance(sparse_pred, dict):
                losses = {
                    key: self.criterion(pred, sparse_gt)
                    for key, pred in sparse_pred.items()
                }
                loss = torch.stack(list(losses.values())).mean()
                for key, key_loss in losses.items():
                    if key not in encoder_losses:
                        encoder_losses[key] = AverageMeter(f'{key} loss')
                    encoder_losses[key].update(key_loss)
            else:
                loss = self.criterion(sparse_pred, sparse_gt)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                msg = (f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                       f'[{batch_idx + 1}/{len(data_loader)}] '
                       f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                       f'MSE loss: {iter_loss.avg:.4e}')
                if encoder_losses:
                    loss_parts = [
                        f'{key}: {meter.avg:.4e}'
                        for key, meter in encoder_losses.items()
                    ]
                    msg += ' | encoders ' + '; '.join(loss_parts)
                msg += f' | time: {iter_time.avg:.3f}'
                logger.info(msg)
                self.vision.add_scalar("every/lr", self.scheduler.get_lr()[0],
                                       global_step=self.cur_epoch)
                self.vision.add_scalar("every/mse_loss", iter_loss.avg, self.cur_epoch)

        mode = 'Train' if self.model.training else 'Val'
        msg = f'=> {mode}  Loss: {iter_loss.avg:.4e}'
        metrics = {"loss": self._as_float(iter_loss.avg)}
        if encoder_losses:
            parts = []
            for key, meter in encoder_losses.items():
                metrics[key] = {"loss": self._as_float(meter.avg)}
                parts.append(f'{key}: loss={meter.avg:.4e}')
            msg += ' | encoders ' + '; '.join(parts)
        logger.info(msg + '\n')
        if self.model.training:
            self.last_train_metrics = metrics
        else:
            self.last_val_metrics = metrics

        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        r""" protected function which resume from checkpoint at the beginning of training.
        """

        if self.resume_file is None:
            return None
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}')
        checkpoint = torch.load(self.resume_file, weights_only=True,
                                map_location=self.device)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_nmse = checkpoint.get('best_nmse',
                                        {'nmse': None, 'epoch': None})
        self.cur_epoch += 1  # start from the next epoch

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                    f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, nmse):
        r""" private function which makes loop() function neater.
        """
        if isinstance(nmse, torch.Tensor):
            nmse = float(nmse.detach().cpu())

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_nmse': self.best_nmse
        }

        # save model with best nmse
        if nmse is not None:
            if self.best_nmse['nmse'] is None or self.best_nmse['nmse'] > nmse:
                self.best_nmse = {'nmse': nmse, 'epoch': self.cur_epoch}
                state['best_nmse'] = self.best_nmse
                self._save(state, name=f"best_nmse.pth")

        # self._save(state, name='last.pth')

        # print current best results
        if self.best_nmse['nmse'] is not None:
            logger.info(f'\n=! Best NMSE: {self.best_nmse["nmse"]:.4e} ('
                        f'epoch={self.best_nmse["epoch"]})\n')
            self.vision.add_scalar("best/mse", self.best_nmse['nmse'],
                                   global_step=self.best_nmse['epoch'])

    def _as_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _write_encoder_scalars(self, ep):
        metrics = getattr(self.tester, "last_metrics", {})
        self.last_test_metrics = metrics
        train_metrics = self.last_train_metrics
        for key, item in metrics.items():
            if key == "aggregate" or not isinstance(item, dict):
                continue
            if "loss" in item:
                self.vision.add_scalar(f"encoders/{key}/test_loss",
                                       item["loss"], global_step=ep)
            if "nmse" in item:
                self.vision.add_scalar(f"encoders/{key}/nmse",
                                       item["nmse"], global_step=ep)
            train_item = train_metrics.get(key, {})
            if isinstance(train_item, dict) and "loss" in train_item:
                self.vision.add_scalar(f"encoders/{key}/train_loss",
                                       train_item["loss"], global_step=ep)

    def _write_adapter_scalars(self, ep):
        if not hasattr(self.model, "adapter_metrics"):
            return
        metrics = self.model.adapter_metrics()
        if not metrics:
            return
        parts = []
        for name, value in metrics.items():
            self.vision.add_scalar(name, value, global_step=ep)
            parts.append(f"{name}={value:.4e}")
        logger.info("=> Adapter metrics: " + " | ".join(parts) + "\n")

    def save_encoder_outputs(self, data_loader, output_path):
        if output_path is None:
            logger.warning('No path to save encoder outputs.')
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        encoder_outputs = []
        sample_indices = []
        with torch.no_grad():
            for batch in data_loader:
                sparse_gt = batch[0]
                indices = batch[1] if len(batch) > 1 else None
                sparse_gt = sparse_gt.to(self.device)
                if hasattr(self.model, 'encoders'):
                    encoder_output = self.model.encode(sparse_gt)
                elif hasattr(self.model, 'encoder'):
                    encoder_output = self.model.encoder(sparse_gt)
                else:
                    encoder_output = self.model.encode(sparse_gt)
                if isinstance(encoder_output, dict):
                    raise TypeError(
                        "save_encoder_outputs cannot save dict outputs; "
                        "use save_all_encoder_outputs for multi-encoder models")
                encoder_outputs.append(encoder_output.cpu())
                if indices is not None:
                    sample_indices.append(indices.cpu())

        encoder_outputs_tensor = torch.cat(encoder_outputs, dim=0)
        index_aligned = len(sample_indices) > 0
        if index_aligned:
            indices_tensor = torch.cat(sample_indices, dim=0).to(torch.long)
            if indices_tensor.numel() != encoder_outputs_tensor.size(0):
                raise ValueError(
                    'number of encoder outputs does not match number of '
                    f'indices: {encoder_outputs_tensor.size(0)} vs '
                    f'{indices_tensor.numel()}')
            if indices_tensor.numel() == 0:
                raise ValueError('cannot save empty indexed encoder outputs')
            expected = torch.arange(indices_tensor.numel(), dtype=torch.long)
            sorted_indices = torch.sort(indices_tensor).values
            if not torch.equal(sorted_indices, expected):
                raise ValueError(
                    'encoder output indices must cover each sample exactly '
                    'once from 0 to N-1')
            aligned_outputs = torch.empty_like(encoder_outputs_tensor)
            aligned_outputs[indices_tensor] = encoder_outputs_tensor
            encoder_outputs_tensor = aligned_outputs

        torch.save(encoder_outputs_tensor, output_path)
        order_msg = 'index-aligned' if index_aligned else 'loader-order'
        logger.info(f'=> Saved {order_msg} encoder outputs '
                    f'{tuple(encoder_outputs_tensor.shape)} to {output_path}')

    def save_all_encoder_outputs(self, loaders, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, 'encoders'):
            for split, data_loader in loaders.items():
                self._save_multi_encoder_outputs(data_loader, output_dir, split)
            return
        for split, data_loader in loaders.items():
            output_path = os.path.join(output_dir, f"{split}_code.pt")
            self.save_encoder_outputs(data_loader, output_path)

    def _save_multi_encoder_outputs(self, data_loader, output_dir, split):
        self.model.eval()
        outputs = {key: [] for key in self.model.encoder_keys}
        sample_indices = []
        with torch.no_grad():
            for batch in data_loader:
                sparse_gt = batch[0].to(self.device)
                indices = batch[1] if len(batch) > 1 else None
                encoder_outputs = self.model.encode(sparse_gt)
                for key, value in encoder_outputs.items():
                    outputs[key].append(value.cpu())
                if indices is not None:
                    sample_indices.append(indices.cpu())

        indices_tensor = None
        index_aligned = len(sample_indices) > 0
        if index_aligned:
            indices_tensor = torch.cat(sample_indices, dim=0).to(torch.long)
            if indices_tensor.numel() == 0:
                raise ValueError('cannot save empty indexed encoder outputs')
            expected = torch.arange(indices_tensor.numel(), dtype=torch.long)
            sorted_indices = torch.sort(indices_tensor).values
            if not torch.equal(sorted_indices, expected):
                raise ValueError(
                    'encoder output indices must cover each sample exactly '
                    'once from 0 to N-1')

        for key, tensors in outputs.items():
            encoder_outputs_tensor = torch.cat(tensors, dim=0)
            if index_aligned:
                if indices_tensor.numel() != encoder_outputs_tensor.size(0):
                    raise ValueError(
                        'number of encoder outputs does not match number of '
                        f'indices: {encoder_outputs_tensor.size(0)} vs '
                        f'{indices_tensor.numel()}')
                aligned_outputs = torch.empty_like(encoder_outputs_tensor)
                aligned_outputs[indices_tensor] = encoder_outputs_tensor
                encoder_outputs_tensor = aligned_outputs
            output_path = os.path.join(output_dir, f"{key}_{split}_code.pt")
            torch.save(encoder_outputs_tensor, output_path)
            order_msg = 'index-aligned' if index_aligned else 'loader-order'
            logger.info(f'=> Saved {order_msg} encoder outputs '
                        f'{tuple(encoder_outputs_tensor.shape)} to {output_path}')



class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.last_metrics = {}

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, nmse = self._iteration(test_data)
        if verbose:
            logger.info(f'\n=> Test result: \nloss: {loss:.4e}'
                        f'    NMSE: {nmse:.4e}\n')
        return loss, nmse

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """

        iter_loss = AverageMeter('Iter loss')
        encoder_losses = {}
        iter_time = AverageMeter('Iter time')
        total_error = torch.tensor(0., device=self.device)
        total_power = torch.tensor(0., device=self.device)
        encoder_errors = {}
        encoder_powers = {}
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0]
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            if isinstance(sparse_pred, dict):
                losses = {}
                for key, pred in sparse_pred.items():
                    key_loss = self.criterion(pred, sparse_gt)
                    losses[key] = key_loss
                    if key not in encoder_losses:
                        encoder_losses[key] = AverageMeter(f'{key} loss')
                        encoder_errors[key] = torch.tensor(
                            0., device=self.device)
                        encoder_powers[key] = torch.tensor(
                            0., device=self.device)
                    encoder_losses[key].update(key_loss)
                    error_sum, power_sum = evaluator(pred, sparse_gt)
                    encoder_errors[key] += error_sum
                    encoder_powers[key] += power_sum
                    total_error += error_sum
                    total_power += power_sum
                loss = torch.stack(list(losses.values())).mean()
            else:
                loss = self.criterion(sparse_pred, sparse_gt)
                error_sum, power_sum = evaluator(sparse_pred, sparse_gt)
                total_error += error_sum
                total_power += power_sum
            nmse = nmse_from_sums(total_error, total_power)

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                msg = (f'[{batch_idx + 1}/{len(data_loader)}] '
                       f'loss: {iter_loss.avg:.4e} | '
                       f'NMSE: {nmse:.4e}')
                if encoder_losses:
                    parts = []
                    for key, meter in encoder_losses.items():
                        key_nmse = nmse_from_sums(encoder_errors[key],
                                                  encoder_powers[key])
                        parts.append(f'{key}: loss={meter.avg:.4e}, '
                                     f'NMSE={key_nmse:.4e}')
                    msg += ' | encoders ' + '; '.join(parts)
                msg += f' | time: {iter_time.avg:.3f}'
                logger.info(msg)

        nmse = nmse_from_sums(total_error, total_power)
        self.last_metrics = {
            "aggregate": {
                "loss": self._as_float(iter_loss.avg),
                "nmse": self._as_float(nmse),
            }
        }
        if encoder_losses:
            parts = []
            for key, meter in encoder_losses.items():
                key_nmse = nmse_from_sums(encoder_errors[key],
                                          encoder_powers[key])
                self.last_metrics[key] = {
                    "loss": self._as_float(meter.avg),
                    "nmse": self._as_float(key_nmse),
                }
                parts.append(f'{key}: loss={meter.avg:.4e}, '
                             f'NMSE={key_nmse:.4e}')
            logger.info(f'=> Test NMSE: {nmse:.4e} | encoders '
                        f'{"; ".join(parts)}\n')
        else:
            logger.info(f'=> Test NMSE: {nmse:.4e}\n')

        return iter_loss.avg, nmse

    def _as_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)
