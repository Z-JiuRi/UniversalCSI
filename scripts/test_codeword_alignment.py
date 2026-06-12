import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.solver import Trainer


class IndexedEncoderModel(nn.Module):
    def forward(self, sparse_gt):
        index = sparse_gt.view(sparse_gt.size(0), -1)[:, 0].to(torch.long)
        return torch.stack(
            (
                index.to(torch.float32),
                index.to(torch.float32) + 1000.0,
            ),
            dim=1,
        )

    def encode(self, sparse_gt):
        return self.forward(sparse_gt)


class IdentityScheduler:
    def step(self):
        pass

    def get_lr(self):
        return [0.0]


class CodeToSparseDecoder(nn.Module):
    def forward(self, code):
        return code[:, 0].view(-1, 1, 1, 1)


class FailingDecoder(nn.Module):
    def forward(self, code):
        raise AssertionError("decoder should not run in code_loss_only mode")


class AdapterPathModel(IndexedEncoderModel):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.decoder = CodeToSparseDecoder()

    def encode(self, sparse_gt):
        return super().encode(sparse_gt) + self.dummy * 0.0


class CodeOnlyPathModel(AdapterPathModel):
    def __init__(self):
        super().__init__()
        self.decoder = FailingDecoder()


class OffsetAdapterModel(IndexedEncoderModel):
    def __init__(self):
        super().__init__()
        self.encoder = IndexedEncoderModel()

    def encode(self, sparse_gt):
        return self.encoder(sparse_gt) + 5000.0


def test_index_aligned_codeword_save():
    num_samples = 17
    data = torch.arange(num_samples, dtype=torch.float32).view(num_samples, 1, 1, 1)
    indices = torch.arange(num_samples, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(data, indices),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(123),
    )
    expected_codes = torch.stack(
        (
            indices.to(torch.float32),
            indices.to(torch.float32) + 1000.0,
        ),
        dim=1,
    )

    trainer = Trainer(
        model=IndexedEncoderModel(),
        device=torch.device("cpu"),
        optimizer=None,
        criterion=nn.MSELoss(),
        scheduler=None,
        tensorboard_dir=tempfile.mkdtemp(prefix="codeword_align_tb_"),
    )

    with tempfile.TemporaryDirectory(prefix="codeword_align_") as tmpdir:
        output_path = os.path.join(tmpdir, "train_code.pt")
        trainer.save_encoder_outputs(loader, output_path)
        saved_codes = torch.load(output_path, weights_only=True)

    assert torch.equal(saved_codes, expected_codes)


def test_rewrite_export_uses_raw_encoder_before_adapter():
    from scripts.rewrite_cost2100_in_codewords import export_codes

    num_samples = 17
    data = torch.arange(num_samples, dtype=torch.float32).view(num_samples, 1, 1, 1)
    indices = torch.arange(num_samples, dtype=torch.long)
    expected_raw_codes = torch.stack(
        (
            indices.to(torch.float32),
            indices.to(torch.float32) + 1000.0,
        ),
        dim=1,
    )
    model = OffsetAdapterModel()
    exported = export_codes(
        model=model,
        train_tensor=data,
        batch_size=4,
        workers=0,
        device=torch.device("cpu"),
        shuffle=True,
    )

    assert torch.equal(exported, expected_raw_codes)
    assert not torch.equal(exported, model.encode(data))


def test_trainer_save_uses_raw_encoder_before_adapter():
    num_samples = 17
    data = torch.arange(num_samples, dtype=torch.float32).view(num_samples, 1, 1, 1)
    indices = torch.arange(num_samples, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(data, indices),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(321),
    )
    expected_raw_codes = torch.stack(
        (
            indices.to(torch.float32),
            indices.to(torch.float32) + 1000.0,
        ),
        dim=1,
    )
    trainer = Trainer(
        model=OffsetAdapterModel(),
        device=torch.device("cpu"),
        optimizer=None,
        criterion=nn.MSELoss(),
        scheduler=None,
        tensorboard_dir=tempfile.mkdtemp(prefix="codeword_align_tb_"),
    )

    with tempfile.TemporaryDirectory(prefix="codeword_align_") as tmpdir:
        output_path = os.path.join(tmpdir, "train_code.pt")
        trainer.save_encoder_outputs(loader, output_path)
        saved_codes = torch.load(output_path, weights_only=True)

    assert torch.equal(saved_codes, expected_raw_codes)


def test_teacher_codes_match_shuffled_batch_indices():
    num_samples = 17
    data = torch.arange(num_samples, dtype=torch.float32).view(num_samples, 1, 1, 1)
    indices = torch.arange(num_samples, dtype=torch.long)
    teacher_codes = torch.stack(
        (
            indices.to(torch.float32),
            indices.to(torch.float32) + 1000.0,
        ),
        dim=1,
    )
    loader = DataLoader(
        TensorDataset(data, indices),
        batch_size=5,
        shuffle=True,
        generator=torch.Generator().manual_seed(456),
    )

    for sparse_gt, batch_indices in loader:
        batch_teacher_codes = teacher_codes[batch_indices]
        batch_expected_codes = torch.stack(
            (
                sparse_gt.view(sparse_gt.size(0), -1)[:, 0],
                sparse_gt.view(sparse_gt.size(0), -1)[:, 0] + 1000.0,
            ),
            dim=1,
        )
        assert torch.equal(batch_teacher_codes, batch_expected_codes)


def test_trainer_code_loss_uses_sample_indices():
    num_samples = 17
    data = torch.arange(num_samples, dtype=torch.float32).view(num_samples, 1, 1, 1)
    indices = torch.arange(num_samples, dtype=torch.long)
    teacher_codes = torch.stack(
        (
            indices.to(torch.float32),
            indices.to(torch.float32) + 1000.0,
        ),
        dim=1,
    )
    loader = DataLoader(
        TensorDataset(data, indices),
        batch_size=5,
        shuffle=True,
        generator=torch.Generator().manual_seed(789),
    )
    model = AdapterPathModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    with tempfile.TemporaryDirectory(prefix="teacher_code_") as tmpdir:
        teacher_code_path = os.path.join(tmpdir, "train_code.pt")
        torch.save(teacher_codes, teacher_code_path)
        trainer = Trainer(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            scheduler=IdentityScheduler(),
            tensorboard_dir=tempfile.mkdtemp(prefix="codeword_align_tb_"),
            teacher_code=teacher_code_path,
            code_loss_lambda=1.0,
            teacher_code_size=num_samples,
        )
        trainer.cur_epoch = 1
        trainer.all_epoch = 1
        loss = trainer.train(loader)

    assert float(loss) == 0.0


def test_trainer_code_loss_only_skips_decoder():
    num_samples = 17
    data = torch.arange(num_samples, dtype=torch.float32).view(num_samples, 1, 1, 1)
    indices = torch.arange(num_samples, dtype=torch.long)
    teacher_codes = torch.stack(
        (
            indices.to(torch.float32),
            indices.to(torch.float32) + 1000.0,
        ),
        dim=1,
    )
    loader = DataLoader(
        TensorDataset(data, indices),
        batch_size=5,
        shuffle=True,
        generator=torch.Generator().manual_seed(987),
    )
    model = CodeOnlyPathModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    with tempfile.TemporaryDirectory(prefix="teacher_code_") as tmpdir:
        teacher_code_path = os.path.join(tmpdir, "train_code.pt")
        torch.save(teacher_codes, teacher_code_path)
        trainer = Trainer(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            scheduler=IdentityScheduler(),
            tensorboard_dir=tempfile.mkdtemp(prefix="codeword_align_tb_"),
            teacher_code=teacher_code_path,
            code_loss_lambda=1.0,
            teacher_code_size=num_samples,
            code_loss_only=True,
        )
        trainer.cur_epoch = 1
        trainer.all_epoch = 1
        loss = trainer.train(loader)

    assert float(loss) == 0.0


if __name__ == "__main__":
    test_index_aligned_codeword_save()
    test_rewrite_export_uses_raw_encoder_before_adapter()
    test_trainer_save_uses_raw_encoder_before_adapter()
    test_teacher_codes_match_shuffled_batch_indices()
    test_trainer_code_loss_uses_sample_indices()
    test_trainer_code_loss_only_skips_decoder()
    print("codeword alignment tests passed")
