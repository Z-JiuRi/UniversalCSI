import time
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from models.canonical_heads import AnchorTargetBuilder
from utils import logger
from utils.statics import AverageMeter, evaluator, nmse_from_sums

__all__ = ['Trainer', 'Tester']

class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', tensorboard_dir=None, print_freq=20,
                 val_freq=10, test_freq=10, test_every_epoch=False,
                 teacher_code_path=None, lambda_recon=1.0, lambda_code=0.0,
                 lambda_fc=0.0, lambda_recT=0.0,
                 anchor_target='none', lambda_anchor=0.0, anchor_loss='mse',
                 train_path=None, channel=2, nt=32, nc=32, cr=4,
                 lambda_code_mean=0.0, lambda_code_var=0.0,
                 lambda_code_cov=0.0, lambda_code_l1=0.0,
                 code_var_tau=256.0):

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

        # Teacher code distillation
        self.lambda_recon = lambda_recon
        self.lambda_code = lambda_code
        self.lambda_fc = lambda_fc
        self.lambda_recT = lambda_recT
        self.teacher_codes = None
        if (lambda_code or lambda_fc or lambda_recT) and teacher_code_path is None:
            raise ValueError(
                "teacher_code is required when lambda_code, lambda_fc, "
                "or lambda_recT is non-zero")
        if teacher_code_path is not None:
            self._load_teacher_codes(teacher_code_path)
            self._move_teacher_codes_to_device()
        self.lambda_anchor = lambda_anchor
        self.anchor_loss = anchor_loss
        self.anchor_target = None
        if anchor_target not in (None, '', 'none') and lambda_anchor > 0:
            code_dim = channel * nt * nc // cr
            self.anchor_target = AnchorTargetBuilder(
                target_type=anchor_target,
                code_dim=code_dim,
                channel=channel,
                nt=nt,
                nc=nc,
                train_path=train_path,
                device=device)
            logger.info(f'=> Enabled {anchor_target} anchor target '
                        f'(lambda={lambda_anchor}, loss={anchor_loss})')

        self.lambda_code_mean = lambda_code_mean
        self.lambda_code_var = lambda_code_var
        self.lambda_code_cov = lambda_code_cov
        self.lambda_code_l1 = lambda_code_l1
        self.code_var_tau = code_var_tau
        self.code_dim = channel * nt * nc // cr
        index = torch.arange(self.code_dim, dtype=torch.float32)
        target_var = torch.exp(-index / code_var_tau)
        target_var = target_var / target_var.mean()
        self.code_target_var = target_var.to(device)

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
        iter_code_loss = AverageMeter('Code loss')
        iter_fc_loss = AverageMeter('FC loss')
        iter_recT_loss = AverageMeter('Teacher recon loss')
        iter_anchor_loss = AverageMeter('Anchor loss')
        iter_code_reg_loss = AverageMeter('Code reg loss')
        iter_recon_loss = AverageMeter('Recon loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0]
            sparse_gt = sparse_gt.to(self.device)
            code_pred = None
            needs_code_pred = (
                self.model.training and (
                    self.teacher_codes is not None
                    or self.anchor_target is not None
                    or self._has_code_regularization()
                )
            )
            if needs_code_pred:
                code_pred = self.model.encode(sparse_gt)
                sparse_pred = self.model.decoder(code_pred)
            else:
                sparse_pred = self.model(sparse_gt)

            # Reconstruction loss (always computed)
            recon_loss = self.criterion(sparse_pred, sparse_gt)
            total_loss = self.lambda_recon * recon_loss

            # Code-space distillation loss (training only, when teacher codes available)
            code_loss = torch.tensor(0., device=self.device)
            fc_loss = torch.tensor(0., device=self.device)
            recT_loss = torch.tensor(0., device=self.device)
            if self.model.training and self.teacher_codes is not None:
                indices = batch[1]
                teacher_code = self.teacher_codes[indices]
                code_loss = self.criterion(code_pred, teacher_code)
                if self.lambda_code:
                    total_loss = total_loss + self.lambda_code * code_loss
                if self.lambda_fc:
                    if not hasattr(self.model.decoder, "fc_decoder"):
                        raise ValueError(
                            "lambda_fc requires decoder.fc_decoder, "
                            f"got {type(self.model.decoder).__name__}")
                    fc_adapter = self.model.decoder.fc_decoder(code_pred)
                    with torch.no_grad():
                        fc_teacher = self.model.decoder.fc_decoder(teacher_code)
                    fc_loss = self.criterion(fc_adapter, fc_teacher)
                    total_loss = total_loss + self.lambda_fc * fc_loss
                if self.lambda_recT:
                    with torch.no_grad():
                        teacher_recon = self.model.decoder(teacher_code)
                    recT_loss = self.criterion(sparse_pred, teacher_recon)
                    total_loss = total_loss + self.lambda_recT * recT_loss

            anchor_loss = torch.tensor(0., device=self.device)
            if self.model.training and self.anchor_target is not None:
                anchor_code = self.anchor_target(sparse_gt)
                if self.anchor_loss == 'cosine':
                    anchor_loss = (
                        1.0 - F.cosine_similarity(
                            F.layer_norm(code_pred, code_pred.shape[1:]),
                            F.layer_norm(anchor_code, anchor_code.shape[1:]),
                            dim=1)).mean()
                else:
                    anchor_loss = self.criterion(
                        F.layer_norm(code_pred, code_pred.shape[1:]),
                        F.layer_norm(anchor_code, anchor_code.shape[1:]))
                total_loss = total_loss + self.lambda_anchor * anchor_loss

            code_reg_loss = torch.tensor(0., device=self.device)
            if self.model.training and self._has_code_regularization():
                code_reg_loss = self._code_regularization_loss(code_pred)
                total_loss = total_loss + code_reg_loss

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(total_loss)
            iter_recon_loss.update(recon_loss)
            if self.model.training and self.teacher_codes is not None:
                iter_code_loss.update(code_loss)
                iter_fc_loss.update(fc_loss)
                iter_recT_loss.update(recT_loss)
            if self.model.training and self.anchor_target is not None:
                iter_anchor_loss.update(anchor_loss)
            if self.model.training and self._has_code_regularization():
                iter_code_reg_loss.update(code_reg_loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                parts = [f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                         f'[{batch_idx + 1}/{len(data_loader)}] '
                         f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                         f'recon: {iter_recon_loss.avg:.4e}']
                if self.model.training and self.teacher_codes is not None:
                    parts.append(f'code: {iter_code_loss.avg:.4e}')
                    if self.lambda_fc:
                        parts.append(f'fc: {iter_fc_loss.avg:.4e}')
                    if self.lambda_recT:
                        parts.append(f'recT: {iter_recT_loss.avg:.4e}')
                if self.model.training and self.anchor_target is not None:
                    parts.append(f'anchor: {iter_anchor_loss.avg:.4e}')
                if self.model.training and self._has_code_regularization():
                    parts.append(f'code_reg: {iter_code_reg_loss.avg:.4e}')
                parts.append(f'total: {iter_loss.avg:.4e}'
                             f' | time: {iter_time.avg:.3f}')
                logger.info(' '.join(parts))
                self.vision.add_scalar("every/lr", self.scheduler.get_lr()[0],
                                       global_step=self.cur_epoch)
                self.vision.add_scalar("every/total_loss", iter_loss.avg,
                                       global_step=self.cur_epoch)

        mode = 'Train' if self.model.training else 'Val'
        parts = [f'=> {mode}  Recon loss: {iter_recon_loss.avg:.4e}']
        if self.model.training and self.teacher_codes is not None:
            parts.append(f'Code loss: {iter_code_loss.avg:.4e}')
            if self.lambda_fc:
                parts.append(f'FC loss: {iter_fc_loss.avg:.4e}')
            if self.lambda_recT:
                parts.append(f'Teacher recon loss: {iter_recT_loss.avg:.4e}')
        if self.model.training and self.anchor_target is not None:
            parts.append(f'Anchor loss: {iter_anchor_loss.avg:.4e}')
        if self.model.training and self._has_code_regularization():
            parts.append(f'Code reg loss: {iter_code_reg_loss.avg:.4e}')
        parts.append(f'Total: {iter_loss.avg:.4e}')
        msg = ' | '.join(parts)
        metrics = {"loss": self._as_float(iter_loss.avg),
                   "recon_loss": self._as_float(iter_recon_loss.avg)}
        if self.model.training and self.teacher_codes is not None:
            metrics["code_loss"] = self._as_float(iter_code_loss.avg)
            if self.lambda_fc:
                metrics["fc_loss"] = self._as_float(iter_fc_loss.avg)
            if self.lambda_recT:
                metrics["teacher_recon_loss"] = self._as_float(iter_recT_loss.avg)
        if self.model.training and self.anchor_target is not None:
            metrics["anchor_loss"] = self._as_float(iter_anchor_loss.avg)
        if self.model.training and self._has_code_regularization():
            metrics["code_reg_loss"] = self._as_float(iter_code_reg_loss.avg)
        adapter_metrics = self._get_adapter_metrics()
        if adapter_metrics:
            parts = [f"{k}={v:.4e}" for k, v in adapter_metrics.items()]
            msg += " | " + " | ".join(parts)
        logger.info(msg + '\n')
        if self.model.training:
            self.last_train_metrics = metrics
        else:
            self.last_val_metrics = metrics

        return iter_loss.avg

    def _has_code_regularization(self):
        return any([
            self.lambda_code_mean,
            self.lambda_code_var,
            self.lambda_code_cov,
            self.lambda_code_l1,
        ])

    def _code_regularization_loss(self, code):
        total = code.new_zeros(())
        if self.lambda_code_mean:
            total = total + self.lambda_code_mean * code.mean(dim=0).pow(2).mean()

        if self.lambda_code_var or self.lambda_code_cov:
            centered = code - code.mean(dim=0, keepdim=True)
            denom = max(code.size(0) - 1, 1)
            cov = centered.t().matmul(centered) / denom
            diag = cov.diag()
            if self.lambda_code_var:
                target = self.code_target_var.to(code.device, code.dtype)
                total = total + self.lambda_code_var * (diag - target).pow(2).mean()
            if self.lambda_code_cov:
                offdiag = cov - torch.diag_embed(diag)
                total = total + self.lambda_code_cov * offdiag.pow(2).mean()

        if self.lambda_code_l1:
            total = total + self.lambda_code_l1 * code.abs().mean()
        return total

    def _load_teacher_codes(self, path):
        r"""Load precomputed teacher codewords from disk."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Teacher code file not found: {path}")
        logger.info(f'=> Loading teacher codewords from {path}')
        self.teacher_codes = torch.load(path, weights_only=True, map_location='cpu')
        if self.teacher_codes.ndim != 2:
            raise ValueError(
                f'Teacher codewords must be 2D (N, code_dim), '
                f'got shape {self.teacher_codes.shape}')
        logger.info(f'=> Loaded teacher codewords: {self.teacher_codes.shape}')

    def _move_teacher_codes_to_device(self):
        r"""Move teacher codes to the compute device for efficient indexing."""
        if self.teacher_codes is not None:
            self.teacher_codes = self.teacher_codes.to(self.device)

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

    def _get_adapter_metrics(self):
        if hasattr(self.model, "adapter_metrics"):
            return self.model.adapter_metrics()
        return {}

    def save_codewords(self, data_loader, output_path):
        if output_path is None:
            logger.warning('No path to save codewords.')
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        codewords = []
        sample_indices = []
        with torch.no_grad():
            for batch in data_loader:
                sparse_gt = batch[0]
                indices = batch[1] if len(batch) > 1 else None
                sparse_gt = sparse_gt.to(self.device)
                codeword = self.model.encode(sparse_gt)
                codewords.append(codeword.cpu())
                if indices is not None:
                    sample_indices.append(indices.cpu())

        codewords_tensor = torch.cat(codewords, dim=0)
        index_aligned = len(sample_indices) > 0
        if index_aligned:
            indices_tensor = torch.cat(sample_indices, dim=0).to(torch.long)
            if indices_tensor.numel() != codewords_tensor.size(0):
                raise ValueError(
                    'number of codewords does not match number of '
                    f'indices: {codewords_tensor.size(0)} vs '
                    f'{indices_tensor.numel()}')
            if indices_tensor.numel() == 0:
                raise ValueError('cannot save empty indexed codewords')
            expected = torch.arange(indices_tensor.numel(), dtype=torch.long)
            sorted_indices = torch.sort(indices_tensor).values
            if not torch.equal(sorted_indices, expected):
                raise ValueError(
                    'codeword indices must cover each sample exactly '
                    'once from 0 to N-1')
            aligned_codewords = torch.empty_like(codewords_tensor)
            aligned_codewords[indices_tensor] = codewords_tensor
            codewords_tensor = aligned_codewords

        torch.save(codewords_tensor, output_path)
        order_msg = 'index-aligned' if index_aligned else 'loader-order'
        logger.info(f'=> Saved {order_msg} codewords '
                    f'{tuple(codewords_tensor.shape)} to {output_path}')



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
                msg = (f'[{batch_idx + 1}/{len(data_loader)}] '
                       f'loss: {iter_loss.avg:.4e} | '
                       f'NMSE: {nmse:.4e} | '
                       f'time: {iter_time.avg:.3f}')
                logger.info(msg)

        nmse = nmse_from_sums(total_error, total_power)
        self.last_metrics = {
            "aggregate": {
                "loss": self._as_float(iter_loss.avg),
                "nmse": self._as_float(nmse),
            }
        }
        msg = f'=> Test NMSE: {nmse:.4e}'
        adapter_metrics = self._get_adapter_metrics()
        if adapter_metrics:
            parts = [f"{k}={v:.4e}" for k, v in adapter_metrics.items()]
            msg += " | " + " | ".join(parts)
        logger.info(msg + '\n')

        return iter_loss.avg, nmse

    def _as_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _get_adapter_metrics(self):
        if hasattr(self.model, "adapter_metrics"):
            return self.model.adapter_metrics()
        return {}
