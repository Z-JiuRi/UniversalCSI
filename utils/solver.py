import time
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from utils import logger
from utils.statics import AverageMeter, evaluator, nmse_from_sums
from models.lora import collect_lora_metrics

__all__ = ['Trainer', 'Tester']

class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', tensorboard_dir=None, print_freq=20,
                 val_freq=10, test_freq=10, lora_training=False,
                 test_every_epoch=False, teacher_code=None,
                 code_loss_lambda=None, teacher_code_size=None,
                 code_loss_raw_lambda=None):

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
        self.lora_training = lora_training
        self.test_every_epoch = test_every_epoch
        self.teacher_code = teacher_code
        self.code_loss_lambda = code_loss_lambda
        self.code_loss_raw_lambda = code_loss_raw_lambda
        self.teacher_codes = self._load_teacher_codes(teacher_code,
                                                       teacher_code_size)

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_nmse = {'nmse': None, 'epoch': None}

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
            else:
                nmse = None

            if self.lora_training:
                self._log_lora_metrics(ep, nmse)

            # conduct saving, visualization and log printing
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
        iter_recon_loss = AverageMeter('Recon loss')
        iter_code_loss = AverageMeter('Code loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        use_code_loss = (
            self.model.training
            and self.teacher_codes is not None
            and (self.code_loss_lambda is not None
                 or self.code_loss_raw_lambda is not None)
        )

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0]
            indices = batch[1] if len(batch) > 1 else None
            sparse_gt = sparse_gt.to(self.device)
            code_loss = None
            if use_code_loss:
                if indices is None:
                    raise ValueError('teacher code loss requires data indices')
                adapted_code = self.model.encode(sparse_gt)
                sparse_pred = self.model.decoder(adapted_code)
                teacher_code = self.teacher_codes[indices.cpu()].to(
                    self.device, non_blocking=True)
                code_loss = self.criterion(adapted_code, teacher_code)
            else:
                sparse_pred = self.model(sparse_gt)

            recon_loss = self.criterion(sparse_pred, sparse_gt)
            loss = recon_loss
            if code_loss is not None:
                loss = loss + self._code_loss_weight() * code_loss

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(loss)
            iter_recon_loss.update(recon_loss)
            if code_loss is not None:
                iter_code_loss.update(code_loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                msg = (f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                       f'[{batch_idx + 1}/{len(data_loader)}] '
                       f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                       f'MSE loss: {iter_loss.avg:.4e}')
                if use_code_loss:
                    code_loss_weight = self._code_loss_weight()
                    msg += (f' | recon: {iter_recon_loss.avg:.4e}'
                            f' | code: {iter_code_loss.avg:.4e}'
                            f' | lambda: {code_loss_weight.item():.4e}')
                msg += f' | time: {iter_time.avg:.3f}'
                logger.info(msg)
                self.vision.add_scalar("every/lr", self.scheduler.get_lr()[0],
                                       global_step=self.cur_epoch)
                self.vision.add_scalar("every/mse_loss", iter_loss.avg, self.cur_epoch)
                if use_code_loss:
                    self.vision.add_scalar("every/recon_loss",
                                           iter_recon_loss.avg, self.cur_epoch)
                    self.vision.add_scalar("every/code_loss",
                                           iter_code_loss.avg, self.cur_epoch)

        mode = 'Train' if self.model.training else 'Val'
        msg = f'=> {mode}  Loss: {iter_loss.avg:.4e}'
        if use_code_loss:
            code_loss_weight = self._code_loss_weight()
            msg += (f' | recon: {iter_recon_loss.avg:.4e}'
                    f' | code: {iter_code_loss.avg:.4e}'
                    f' | lambda: {code_loss_weight.item():.4e}')
        logger.info(msg + '\n')

        return iter_loss.avg

    def _code_loss_weight(self):
        if self.code_loss_raw_lambda is not None:
            return F.softplus(self.code_loss_raw_lambda)
        return torch.tensor(float(self.code_loss_lambda), device=self.device)

    def _load_teacher_codes(self, teacher_code, teacher_code_size):
        if teacher_code is None:
            return None
        if not os.path.isfile(teacher_code):
            raise FileNotFoundError(teacher_code)
        teacher_codes = torch.load(teacher_code, weights_only=True,
                                   map_location=torch.device('cpu'))
        teacher_codes = teacher_codes.to(torch.float32)
        if teacher_codes.ndim != 2:
            raise ValueError(
                f'teacher codes should have shape (N, code_dim), '
                f'got {tuple(teacher_codes.shape)}')
        if (teacher_code_size is not None
                and teacher_codes.size(0) != teacher_code_size):
            raise ValueError(
                f'teacher code count {teacher_codes.size(0)} does not match '
                f'train dataset size {teacher_code_size}')
        logger.info(f'=> Teacher codes loaded from {teacher_code}: '
                    f'{tuple(teacher_codes.shape)}; '
                    f'code_loss_lambda={self.code_loss_lambda}; '
                    f'learnable_code_loss_lambda='
                    f'{self.code_loss_raw_lambda is not None}')
        return teacher_codes

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
        raw_lambda = checkpoint.get('code_loss_raw_lambda')
        if raw_lambda is not None and self.code_loss_raw_lambda is not None:
            self.code_loss_raw_lambda.data.copy_(raw_lambda.to(self.device))
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
        if self.code_loss_raw_lambda is not None:
            state['code_loss_raw_lambda'] = (
                self.code_loss_raw_lambda.detach().cpu())

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

    def _log_lora_metrics(self, epoch, nmse):
        metrics = collect_lora_metrics(self.model)
        if not metrics:
            logger.warning("LoRA training enabled but no LoRA metrics found.")
            return
        fields = []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, float):
                fields.append(f"{key}: {value:.4e}")
                self.vision.add_scalar(f"lora/{key}", value, global_step=epoch)
            else:
                fields.append(f"{key}: {value}")
        if isinstance(nmse, torch.Tensor):
            nmse = float(nmse.detach().cpu())
        nmse_msg = "None" if nmse is None else f"{nmse:.4e}"
        logger.info(f"=> LoRA Epoch {epoch}: NMSE={nmse_msg} | "
                    + " | ".join(fields))

    def save_encoder_outputs(self, data_loader, output_path):
        if output_path is None:
            logger.warning('No path to save encoder outputs.')
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        encoder_outputs = []
        with torch.no_grad():
            for batch in data_loader:
                sparse_gt = batch[0]
                sparse_gt = sparse_gt.to(self.device)
                encoder_output = self.model.encode(sparse_gt)
                encoder_outputs.append(encoder_output.cpu())

        encoder_outputs_tensor = torch.cat(encoder_outputs, dim=0)
        torch.save(encoder_outputs_tensor, output_path)
        logger.info(f'=> Saved encoder outputs {tuple(encoder_outputs_tensor.shape)} '
                    f'to {output_path}')

    def save_all_encoder_outputs(self, loaders, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for split, data_loader in loaders.items():
            output_path = os.path.join(output_dir, f"{split}_code.pt")
            self.save_encoder_outputs(data_loader, output_path)



class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

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
        iter_time = AverageMeter('Iter time')
        total_error = torch.tensor(0., device=self.device)
        total_power = torch.tensor(0., device=self.device)
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0]
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
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
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.4e} | '
                            f'NMSE: {nmse:.4e} | time: {iter_time.avg:.3f}')

        nmse = nmse_from_sums(total_error, total_power)
        logger.info(f'=> Test NMSE: {nmse:.4e}\n')


        return iter_loss.avg, nmse
